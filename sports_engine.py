#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞彩预测算法引擎 v1.9
支持足球和篮球预测，多维度分析，综合加权评分
v1.9 新增冷门概率模型：基于实力差距、状态波动、联赛冷门系数、伤停不确定性、H2H波动性综合估算冷门概率，并校准胜平负概率分布
"""

import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


VERSION = "v1.9"


# ============================================================
# 维度权重配置
# ============================================================
FOOTBALL_WEIGHTS = {
    "ranking": 0.12,        # 球队排名（v1.8 进一步降低权重）
    "head_to_head": 0.20,   # 历史交锋（v1.8 提升权重，关键因素）
    "recent_form": 0.28,    # 近期状态（v1.8 持续提升权重，最关键因素）
    "home_away": 0.15,      # 主客场优势
    "injuries": 0.15,       # 球员伤停
    "schedule_density": 0.08,  # 赛程密集度
    "weather": 0.02,        # 天气因素（v1.8 进一步降低权重）
}

BASKETBALL_WEIGHTS = {
    "ranking": 0.12,        # 球队排名（v1.8 降低权重）
    "head_to_head": 0.17,   # 历史交锋（v1.8 提升权重）
    "recent_form": 0.25,    # 近期状态（v1.8 提升权重）
    "home_away": 0.12,      # 主客场优势
    "injuries": 0.18,       # 球员伤停
    "schedule_density": 0.12,  # 赛程密集度
    "weather": 0.04,        # 天气因素（v1.8 降低权重）
}


# ============================================================
# 联赛冷门系数配置（v1.9 新增）
# 基于历史冷门频率统计，值越高表示该联赛冷门越常见
# ============================================================
LEAGUE_COLD_FACTOR = {
    # 足球联赛
    "英超": 0.22, "西甲": 0.20, "德甲": 0.18, "意甲": 0.20, "法甲": 0.24,
    "荷甲": 0.22, "葡超": 0.25, "瑞超": 0.24, "挪超": 0.26, "美职联": 0.30,
    "中超": 0.25, "J联赛": 0.25, "K联赛": 0.22, "日职": 0.28, "澳超": 0.28,
    "俄超": 0.23, "土超": 0.26, "比甲": 0.23, "奥甲": 0.22, "丹超": 0.24,
    "苏超": 0.22, "欧冠": 0.18, "欧联": 0.22, "欧协联": 0.24, "世界杯": 0.20,
    "欧洲杯": 0.18, "美洲杯": 0.22, "亚冠": 0.25,
    # 篮球联赛
    "NBA": 0.28, "CBA": 0.25, "Euro": 0.20, "WNBA": 0.25,
    "NBL": 0.26, "VTB": 0.22, "ACB": 0.20,
}
DEFAULT_COLD_FACTOR = 0.22  # 默认冷门系数


# ============================================================
# 工具函数
# ============================================================
def sigmoid(x: float) -> float:
    """Sigmoid 函数，用于将评分映射到 0-1 区间"""
    return 1.0 / (1.0 + math.exp(-x))


def normalize_score(score: float, scale: float = 1.0) -> float:
    """将评分归一化到 0-1 区间"""
    return max(0.0, min(1.0, (score + scale) / (2 * scale)))


# ============================================================
# 各维度评分计算
# ============================================================
class DimensionScorer:
    """各维度评分计算器"""

    @staticmethod
    def ranking_score(home_rank: int, away_rank: int, total_teams: int = 20) -> float:
        """
        球队排名维度评分
        返回主队相对优势分（-1 到 1），正值表示主队占优
        """
        rank_diff = away_rank - home_rank  # 排名数字越小越好，所以用 away - home
        max_diff = total_teams - 1
        normalized = rank_diff / max_diff  # 归一化到 -1 到 1
        return max(-1.0, min(1.0, normalized))

    @staticmethod
    def head_to_head_score(h2h_records: List[Dict]) -> float:
        """
        历史交锋维度评分
        h2h_records: 历史交锋记录列表，每项包含 result ('win'/'draw'/'lose' for home)
        返回主队相对优势分（-1 到 1）
        """
        if not h2h_records:
            return 0.0

        wins = sum(1 for r in h2h_records if r.get("result") == "win")
        draws = sum(1 for r in h2h_records if r.get("result") == "draw")
        losses = sum(1 for r in h2h_records if r.get("result") == "lose")
        total = len(h2h_records)

        # 加权计算：胜=+1，平=0，负=-1
        score = (wins - losses) / total
        # 平局给予轻微的主场优势修正
        score += draws * 0.1 / total
        return max(-1.0, min(1.0, score))

    @staticmethod
    def recent_form_score(home_form: List[str], away_form: List[str]) -> float:
        """
        近期状态维度评分
        form: 最近 N 场比赛结果列表 ['W','D','L',...]
        返回主队相对优势分（-1 到 1）
        """
        def form_to_score(form_list):
            if not form_list:
                return 0.0
            score_map = {"W": 1.0, "D": 0.3, "L": -1.0}
            # 越近的比赛权重越高
            n = len(form_list)
            total_weight = sum(range(1, n + 1))
            weighted = sum(
                score_map.get(r, 0.0) * (i + 1)
                for i, r in enumerate(reversed(form_list))
            )
            return weighted / total_weight

        home_score = form_to_score(home_form)
        away_score = form_to_score(away_form)
        diff = home_score - away_score
        return max(-1.0, min(1.0, diff))

    @staticmethod
    def home_away_score(
        home_team_home_record: Dict,
        away_team_away_record: Dict,
        sport: str = "football"
    ) -> float:
        """
        主客场优势维度评分
        home_team_home_record: 主队主场战绩 {"wins": 5, "draws": 3, "losses": 2, "games": 10}
        away_team_away_record: 客队客场战绩
        返回主队相对优势分（-1 到 1）
        """
        def record_rate(record, sport_type):
            games = record.get("games", 0)
            if games == 0:
                return 0.0
            wins = record.get("wins", 0)
            draws = record.get("draws", 0)
            losses = record.get("losses", 0)
            if sport_type == "football":
                # 足球：胜=3分，平=1分，负=0分
                points = wins * 3 + draws * 1
                max_points = games * 3
                return points / max_points
            else:
                # 篮球：胜率
                return wins / games

        home_rate = record_rate(home_team_home_record, sport)
        away_rate = record_rate(away_team_away_record, sport)
        diff = home_rate - away_rate

        # 主场加成系数
        home_bonus = 0.15 if sport == "football" else 0.1
        diff += home_bonus
        return max(-1.0, min(1.0, diff))

    @staticmethod
    def injuries_score(
        home_injuries: List[Dict],
        away_injuries: List[Dict],
        sport: str = "football"
    ) -> float:
        """
        球员伤停维度评分
        injuries: 伤停球员列表，每项包含 name, position, importance(1-10)
        返回主队相对优势分（-1 到 1），正值表示主队伤停更少/更轻
        """
        def injury_impact(injury_list):
            if not injury_list:
                return 0.0
            total_impact = sum(p.get("importance", 5) for p in injury_list)
            # 基准：足球25人阵容，篮球12人阵容，核心球员总重要性约为 50/30
            baseline = 50.0 if sport == "football" else 30.0
            return min(1.0, total_impact / baseline)

        home_impact = injury_impact(home_injuries)
        away_impact = injury_impact(away_injuries)
        # 客队伤停越重，主队越占优
        diff = away_impact - home_impact
        return max(-1.0, min(1.0, diff))

    @staticmethod
    def schedule_density_score(
        home_last_match_days: int,
        away_last_match_days: int,
        upcoming_matches_days: int = 7,
        home_upcoming: int = 0,
        away_upcoming: int = 0,
        sport: str = "football"
    ) -> float:
        """
        赛程密集度维度评分
        last_match_days: 距离上一场比赛的天数
        upcoming: 未来 N 天内的比赛场数
        返回主队相对优势分（-1 到 1）
        """
        # 休息天数评分：休息越多恢复越好，但过多休息也可能状态下滑
        def rest_score(days):
            optimal = 5 if sport == "football" else 2
            if days <= 0:
                return -1.0
            if days <= optimal:
                return (days / optimal) * 2 - 1  # -1 到 1
            else:
                # 超过最优休息天数，缓慢下降
                excess = days - optimal
                decay = math.exp(-excess / 10)
                return decay

        home_rest = rest_score(home_last_match_days)
        away_rest = rest_score(away_last_match_days)

        # 未来赛程压力
        def upcoming_pressure(count, days):
            if days <= 0:
                return 0.0
            baseline = 2 if sport == "football" else 3  # 基准比赛数
            return min(1.0, count / baseline) * 0.3

        home_pressure = upcoming_pressure(home_upcoming, upcoming_matches_days)
        away_pressure = upcoming_pressure(away_upcoming, upcoming_matches_days)

        # 综合：休息优势 + 赛程压力优势
        rest_diff = home_rest - away_rest
        pressure_diff = away_pressure - home_pressure
        total = rest_diff * 0.7 + pressure_diff * 0.3

        return max(-1.0, min(1.0, total))

    @staticmethod
    def weather_score(weather: Dict, sport: str = "football") -> float:
        """
        天气因素维度评分
        weather: {"temperature": 20, "rainfall": 0, "wind_speed": 5, "condition": "sunny"}
        返回对主队的影响评分（-1 到 1）
        假设主队更适应主场天气，极端天气对技术型球队不利
        """
        if sport == "basketball":
            # 篮球多为室内场馆，天气影响极小
            return 0.05

        if not weather:
            return 0.0

        temp = weather.get("temperature", 20)
        rainfall = weather.get("rainfall", 0)
        wind = weather.get("wind_speed", 5)

        # 温度评分（最佳 15-25 度）
        if 15 <= temp <= 25:
            temp_score = 1.0
        elif temp < 15:
            temp_score = max(-1.0, (temp - 15) / 15)
        else:
            temp_score = max(-1.0, (25 - temp) / 15)

        # 降雨评分
        rain_score = max(-1.0, 1.0 - rainfall / 20.0)

        # 风速评分
        wind_score = max(-1.0, 1.0 - wind / 30.0)

        # 综合天气评分（对主队略有主场适应性加成）
        total = temp_score * 0.3 + rain_score * 0.4 + wind_score * 0.3
        home_adaptation_bonus = 0.1  # 主队更适应主场天气
        total = total * 0.5 + home_adaptation_bonus

        return max(-1.0, min(1.0, total))


# ============================================================
# 比赛预测器
# ============================================================
class MatchPredictor:
    """单场比赛预测器"""

    def __init__(self, sport_type: str):
        self.sport_type = sport_type.lower()
        if self.sport_type == "football":
            self.weights = FOOTBALL_WEIGHTS
        elif self.sport_type == "basketball":
            self.weights = BASKETBALL_WEIGHTS
        else:
            raise ValueError(f"不支持的运动类型: {sport_type}")
        self.scorer = DimensionScorer()

    def _calculate_upset_probability(self, match_data: Dict) -> Dict:
        """
        计算冷门概率（v1.9 优化版）

        基于多个维度综合估算冷门发生的可能性：
        1. 实力差距因子：双方排名差越大，冷门概率越低（sigmoid转换，增强区分度）
        2. 状态波动性：使用信息熵衡量球队近期结果的不可预测性
        3. 联赛冷门系数：不同联赛历史冷门频率不同
        4. 伤停不确定性：关键球员伤停增加不确定性
        5. H2H波动性：历史交锋结果越不稳定，冷门概率越高

        使用 sigmoid(raw_score) 将加权和压缩到合理范围，避免大面积触及上限
        """
        import math

        # 1. 实力差距因子（sigmoid转换，增强区分度）
        home_rank = match_data.get("home_rank", 10)
        away_rank = match_data.get("away_rank", 10)
        total_teams = match_data.get("total_teams", 20)
        rank_gap = abs(home_rank - away_rank) / max(total_teams, 1)
        # 使用 sigmoid 映射：rank_gap 接近0时 uncertainty 高，接近1时 uncertainty 低
        # 转换点设在 rank_gap=0.5 处，斜率 6 使区分更明显
        strength_uncertainty = 1.0 - (1.0 / (1.0 + math.exp(-6 * (rank_gap - 0.5))))

        # 2. 状态波动性（使用信息熵衡量）
        home_form = match_data.get("home_form", [])
        away_form = match_data.get("away_form", [])

        def form_entropy(form_list):
            """使用信息熵衡量状态不可预测性"""
            if len(form_list) < 3:
                return 0.3
            total = len(form_list)
            w_count = form_list.count("W")
            d_count = form_list.count("D")
            l_count = form_list.count("L")
            probs = [c / total for c in [w_count, d_count, l_count] if c > 0]
            # 信息熵：H = -sum(p * log2(p))
            entropy = -sum(p * math.log2(p) for p in probs)
            # 归一化：max entropy for 3 outcomes = log2(3) ≈ 1.585
            return min(1.0, entropy / math.log2(3))

        form_uncertainty = (form_entropy(home_form) + form_entropy(away_form)) / 2

        # 3. 联赛冷门系数
        league = match_data.get("league", "")
        cold_factor = DEFAULT_COLD_FACTOR
        for league_key, cf in LEAGUE_COLD_FACTOR.items():
            if league_key in league or league in league_key:
                cold_factor = cf
                break

        # 4. 伤停不确定性
        home_injuries = match_data.get("home_injuries", [])
        away_injuries = match_data.get("away_injuries", [])
        total_injuries = len(home_injuries) + len(away_injuries)
        all_importance = [p.get("importance", 5) for p in home_injuries] + \
                         [p.get("importance", 5) for p in away_injuries]
        injury_severity = sum(all_importance) / max(len(all_importance), 1) / 10.0
        injury_uncertainty = min(0.6, (total_injuries / 5.0) * (0.3 + 0.7 * injury_severity))

        # 5. H2H波动性（使用信息熵）
        h2h = match_data.get("h2h", [])
        h2h_volatility = 0.2
        if len(h2h) >= 3:
            total = len(h2h)
            results = [r.get("result", "") for r in h2h]
            w_count = results.count("win")
            d_count = results.count("draw")
            l_count = results.count("lose")
            probs = [c / total for c in [w_count, d_count, l_count] if c > 0]
            entropy = -sum(p * math.log2(p) for p in probs)
            h2h_volatility = min(1.0, entropy / math.log2(3))

        # 综合冷门概率——使用 sigmoid 将加权和压缩到合理范围
        # 权重：strength 0.30, form 0.18, cold 0.20, injury 0.15, h2h 0.17
        raw_score = (
            strength_uncertainty * 0.30 +
            form_uncertainty * 0.18 +
            cold_factor * 0.20 +
            injury_uncertainty * 0.15 +
            h2h_volatility * 0.17
        )

        # 使用 sigmoid 将 raw_score 映射到合理范围
        # 偏移量 -0.25 确保中等风险场景的冷门概率在 15-20% 左右
        scaled = 1.0 / (1.0 + math.exp(-6 * (raw_score - 0.25)))
        upset_prob = 0.05 + scaled * 0.25  # 映射到 5%-30% 范围

        return {
            "upset_probability": round(upset_prob, 4),
            "strength_uncertainty": round(strength_uncertainty, 4),
            "form_volatility": round(form_uncertainty, 4),
            "league_cold_factor": cold_factor,
            "injury_uncertainty": round(injury_uncertainty, 4),
            "h2h_volatility": round(h2h_volatility, 4),
        }

    def _calibrate_probabilities(
        self, home_win_prob: float, draw_prob: float, upset_info: Dict
    ) -> Tuple[float, float, float]:
        """
        根据冷门概率校准胜平负概率（v1.9 新增）

        冷门概率主要从高置信度方向分流：
        - 主胜概率越高，冷门意味着主队翻车，分流向客胜和平局
        - 客胜概率越高，冷门意味着客队翻车，分流向主胜和平局
        """
        upset_prob = upset_info["upset_probability"]

        # 判断当前推荐方向，计算冷门冲击强度
        if home_win_prob >= 0.5:
            # 看好主队：冷门从主胜分流
            confidence = home_win_prob
            upset_impact = upset_prob * (0.3 + 0.4 * confidence)
            calibrated_home = home_win_prob * (1 - upset_impact)
            upset_to_away = upset_impact * 0.55
            upset_to_draw = upset_impact * 0.45
            calibrated_draw = draw_prob * (1 - upset_impact * 0.3) + upset_to_draw
            calibrated_away = (1 - home_win_prob - draw_prob) * (1 - upset_impact * 0.3) + upset_to_away
        else:
            # 看好客队：冷门从客胜分流
            away_win_prob = 1.0 - home_win_prob
            confidence = away_win_prob
            upset_impact = upset_prob * (0.3 + 0.4 * confidence)
            calibrated_away = away_win_prob * (1 - upset_impact)
            upset_to_home = upset_impact * 0.55
            upset_to_draw = upset_impact * 0.45
            calibrated_draw = draw_prob * (1 - upset_impact * 0.3) + upset_to_draw
            calibrated_home = home_win_prob * (1 - upset_impact * 0.3) + upset_to_home

        # 归一化
        total = calibrated_home + calibrated_draw + calibrated_away
        if total > 0:
            calibrated_home /= total
            calibrated_draw /= total
            calibrated_away /= total

        return calibrated_home, calibrated_draw, calibrated_away

    def predict(self, match_data: Dict) -> Dict:
        """
        预测单场比赛

        match_data 结构:
        {
            "match_id": "xxx",
            "league": "英超",
            "home_team": "曼城",
            "away_team": "利物浦",
            "home_rank": 1,
            "away_rank": 2,
            "total_teams": 20,
            "h2h": [{"result": "win"}, ...],
            "home_form": ["W", "W", "D", "L", "W"],
            "away_form": ["W", "D", "W", "W", "L"],
            "home_home_record": {"wins": 8, "draws": 2, "losses": 0, "games": 10},
            "away_away_record": {"wins": 5, "draws": 3, "losses": 2, "games": 10},
            "home_injuries": [{"name": "xxx", "position": "MF", "importance": 8}, ...],
            "away_injuries": [...],
            "home_last_match_days": 3,
            "away_last_match_days": 5,
            "home_upcoming": 1,
            "away_upcoming": 2,
            "weather": {"temperature": 22, "rainfall": 0, "wind_speed": 5, "condition": "sunny"}
        }
        """
        # 计算各维度评分
        dimension_scores = {}

        # 1. 球队排名
        dimension_scores["ranking"] = self.scorer.ranking_score(
            match_data.get("home_rank", 10),
            match_data.get("away_rank", 10),
            match_data.get("total_teams", 20)
        )

        # 2. 历史交锋
        dimension_scores["head_to_head"] = self.scorer.head_to_head_score(
            match_data.get("h2h", [])
        )

        # 3. 近期状态
        dimension_scores["recent_form"] = self.scorer.recent_form_score(
            match_data.get("home_form", []),
            match_data.get("away_form", [])
        )

        # 4. 主客场优势
        dimension_scores["home_away"] = self.scorer.home_away_score(
            match_data.get("home_home_record", {}),
            match_data.get("away_away_record", {}),
            self.sport_type
        )

        # 5. 球员伤停
        dimension_scores["injuries"] = self.scorer.injuries_score(
            match_data.get("home_injuries", []),
            match_data.get("away_injuries", []),
            self.sport_type
        )

        # 6. 赛程密集度
        dimension_scores["schedule_density"] = self.scorer.schedule_density_score(
            match_data.get("home_last_match_days", 5),
            match_data.get("away_last_match_days", 5),
            home_upcoming=match_data.get("home_upcoming", 0),
            away_upcoming=match_data.get("away_upcoming", 0),
            sport=self.sport_type
        )

        # 7. 天气因素
        dimension_scores["weather"] = self.scorer.weather_score(
            match_data.get("weather", {}),
            self.sport_type
        )

        # 加权综合评分
        total_score = 0.0
        for dim, score in dimension_scores.items():
            weight = self.weights.get(dim, 0)
            total_score += score * weight

        # 转换为主队胜率
        home_win_prob = normalize_score(total_score, scale=0.8)

        # 计算冷门概率（v1.9 新增）
        upset_info = self._calculate_upset_probability(match_data)

        # 生成预测结果（含冷门校准）
        result = self._build_prediction(
            match_data, dimension_scores, total_score, home_win_prob, upset_info
        )

        return result

    def _build_prediction(
        self,
        match_data: Dict,
        dimension_scores: Dict,
        total_score: float,
        home_win_prob: float,
        upset_info: Dict = None  # v1.9 新增冷门信息
    ) -> Dict:
        """构建预测结果，包含多种玩法（v1.9 新增冷门概率校准）"""
        if upset_info is None:
            upset_info = {"upset_probability": 0.3}

        away_win_prob = 1.0 - home_win_prob

        if self.sport_type == "football":
            # 足球：胜平负
            draw_prob = 0.25  # 基础平局概率
            # 根据实力差距调整平局概率
            strength_diff = abs(home_win_prob - 0.5) * 2  # 0 到 1
            draw_prob = draw_prob * (1 - strength_diff * 0.6)

            # 先做基础概率分配（确保非负且总和为1）
            non_draw = 1.0 - draw_prob
            base_home = home_win_prob * non_draw
            base_away = (1 - home_win_prob) * non_draw

            # 冷门概率校准（v1.9）- 在基础概率上校准
            calibrated_home, calibrated_draw, calibrated_away = self._calibrate_probabilities(
                base_home, draw_prob, upset_info
            )

            # 推荐结果（胜平负）- 使用校准后的概率
            max_prob = max(calibrated_home, calibrated_draw, calibrated_away)
            if max_prob == calibrated_home:
                recommendation = "主胜"
            elif max_prob == calibrated_away:
                recommendation = "客胜"
            else:
                recommendation = "平局"

            # 校准后的信心指数（考虑冷门不确定性）
            raw_confidence = max_prob
            # 冷门概率越高，信心越要打折（v1.9 降幅从0.3调整为0.2）
            confidence = raw_confidence * (1.0 - upset_info["upset_probability"] * 0.2)

            # 预测比分
            predicted_score = self._predict_football_score(
                calibrated_home, calibrated_draw, calibrated_away, match_data
            )

            # 让球胜平负预测
            handicap = self._predict_football_handicap(
                calibrated_home, calibrated_draw, calibrated_away, match_data
            )

            # 大小球预测
            over_under = self._predict_football_over_under(
                calibrated_home, calibrated_away, match_data
            )

            prediction = {
                "match_id": match_data.get("match_id", ""),
                "league": match_data.get("league", ""),
                "home_team": match_data.get("home_team", ""),
                "away_team": match_data.get("away_team", ""),
                "sport_type": "football",
                "dimension_scores": {k: round(v, 4) for k, v in dimension_scores.items()},
                "total_score": round(total_score, 4),
                # 玩法1: 胜平负（校准后）
                "probabilities": {
                    "home_win": round(calibrated_home, 4),
                    "draw": round(calibrated_draw, 4),
                    "away_win": round(calibrated_away, 4)
                },
                "recommendation": recommendation,
                "confidence": round(confidence, 4),
                # 玩法2: 比分预测
                "predicted_score": predicted_score,
                # 玩法3: 让球胜平负
                "handicap": handicap,
                # 玩法4: 大小球
                "over_under": over_under,
                # v1.9 新增冷门信息
                "upset_analysis": {
                    "upset_probability": upset_info["upset_probability"],
                    "strength_uncertainty": upset_info.get("strength_uncertainty", 0),
                    "form_volatility": upset_info.get("form_volatility", 0),
                    "league_cold_factor": upset_info.get("league_cold_factor", 0.3),
                    "injury_uncertainty": upset_info.get("injury_uncertainty", 0),
                    "h2h_volatility": upset_info.get("h2h_volatility", 0),
                }
            }
        else:
            # 篮球：胜负
            # 先做基础概率分配
            base_home = home_win_prob
            base_away = 1 - home_win_prob
            base_draw = 0.0  # 篮球无平局

            # 冷门概率校准（v1.9）
            calibrated_home, _, calibrated_away = self._calibrate_probabilities(
                base_home, base_draw, upset_info
            )

            if calibrated_home >= 0.5:
                recommendation = "主胜"
            else:
                recommendation = "客胜"

            # 校准后的信心指数（篮球）
            raw_confidence = max(calibrated_home, calibrated_away)
            confidence = raw_confidence * (1.0 - upset_info["upset_probability"] * 0.2)

            # 让分预测（篮球）
            spread = self._predict_basketball_spread(calibrated_home, match_data)

            # 大小分预测
            total_points = self._predict_total_points(match_data)

            # 让分胜负预测
            spread_recommendation = "主让胜" if spread > 0 else "客让胜"

            prediction = {
                "match_id": match_data.get("match_id", ""),
                "league": match_data.get("league", ""),
                "home_team": match_data.get("home_team", ""),
                "away_team": match_data.get("away_team", ""),
                "sport_type": "basketball",
                "dimension_scores": {k: round(v, 4) for k, v in dimension_scores.items()},
                "total_score": round(total_score, 4),
                # 玩法1: 胜负（校准后）
                "probabilities": {
                    "home_win": round(calibrated_home, 4),
                    "away_win": round(calibrated_away, 4)
                },
                "recommendation": recommendation,
                "confidence": round(confidence, 4),
                # 玩法2: 让分
                "spread": round(spread, 1),
                "spread_recommendation": spread_recommendation,
                # 玩法3: 大小分
                "total_points": round(total_points, 1),
                "over_under": "大分" if calibrated_home > 0.5 else "小分",
                "over_under_confidence": round(abs(calibrated_home - 0.5) * 2, 4),
                # v1.9 新增冷门信息
                "upset_analysis": {
                    "upset_probability": upset_info["upset_probability"],
                    "strength_uncertainty": upset_info.get("strength_uncertainty", 0),
                    "form_volatility": upset_info.get("form_volatility", 0),
                    "league_cold_factor": upset_info.get("league_cold_factor", 0.3),
                    "injury_uncertainty": upset_info.get("injury_uncertainty", 0),
                    "h2h_volatility": upset_info.get("h2h_volatility", 0),
                }
            }

        return prediction

    def _predict_football_score(
        self, home_prob: float, draw_prob: float, away_prob: float, match_data: Dict
    ) -> str:
        """预测足球比分（改进模型）"""
        # 基于概率估算期望进球数
        # 根据联赛调整进球基数
        league = match_data.get("league", "")
        base_goals = 1.3
        if any(k in league for k in ["英超", "德甲", "荷甲", "美职联"]):
            base_goals = 1.5
        elif any(k in league for k in ["法甲", "意甲", "中超"]):
            base_goals = 1.2
        elif any(k in league for k in ["瑞超", "挪超"]):
            base_goals = 1.4

        home_goals = base_goals * (1 + home_prob * 0.6)
        away_goals = base_goals * (1 + away_prob * 0.6)

        # 引入随机抖动，避免全部相同
        import random
        seed = hash(match_data.get("match_id", "")) % 1000
        rng = random.Random(seed)
        home_jitter = rng.uniform(-0.3, 0.3)
        away_jitter = rng.uniform(-0.3, 0.3)

        home_int = max(0, int(round(home_goals + home_jitter)))
        away_int = max(0, int(round(away_goals + away_jitter)))

        # 平局调整：如果平局概率高，比分应接近
        if draw_prob > 0.18 and abs(home_int - away_int) <= 1:
            # 已经接近，不需要调整
            pass
        elif draw_prob > 0.25:
            # 高平局概率，让比分更接近
            if home_int > away_int:
                home_int = max(0, home_int - 1)
            elif away_int > home_int:
                away_int = max(0, away_int - 1)

        return f"{home_int}-{away_int}"

    def _predict_football_handicap(
        self, home_prob: float, draw_prob: float, away_prob: float, match_data: Dict
    ) -> Dict:
        """预测足球让球胜平负"""
        # 计算让球数（基于主胜概率）
        # 胜率60%≈让0.5球，胜率70%≈让1球
        prob_diff = home_prob - away_prob
        handicap_value = round(prob_diff * 2.5, 1)

        # 限制让球范围 -1.5 到 +1.5
        handicap_value = max(-1.5, min(1.5, handicap_value))

        # 根据让球后的概率分布
        # 让球后主队胜率 = 原主队胜率 + 让球数 * 调整系数
        adjusted_home = home_prob + handicap_value * 0.12
        adjusted_away = away_prob - handicap_value * 0.12
        adjusted_draw = 1.0 - adjusted_home - adjusted_away

        # 归一化
        total_adj = adjusted_home + adjusted_draw + adjusted_away
        adjusted_home /= total_adj
        adjusted_draw /= total_adj
        adjusted_away /= total_adj

        # 推荐让球结果
        max_adj = max(adjusted_home, adjusted_draw, adjusted_away)
        if max_adj == adjusted_home:
            handicap_recommendation = f"主让{handicap_value:.1f}胜"
        elif max_adj == adjusted_away:
            handicap_recommendation = f"客让{abs(handicap_value):.1f}胜"
        else:
            handicap_recommendation = f"{'主' if handicap_value >= 0 else '客'}让{abs(handicap_value):.1f}平"

        return {
            "handicap_value": handicap_value,
            "probabilities": {
                "home_win": round(adjusted_home, 4),
                "draw": round(adjusted_draw, 4),
                "away_win": round(adjusted_away, 4),
            },
            "recommendation": handicap_recommendation,
            "confidence": round(max_adj, 4),
        }

    def _predict_football_over_under(
        self, home_prob: float, away_prob: float, match_data: Dict
    ) -> Dict:
        """预测足球大小球"""
        league = match_data.get("league", "")
        # 根据联赛设定大小球线
        if any(k in league for k in ["英超", "德甲", "荷甲", "美职联"]):
            ou_line = 2.75
        elif any(k in league for k in ["法甲", "意甲", "中超"]):
            ou_line = 2.25
        elif any(k in league for k in ["瑞超", "挪超"]):
            ou_line = 2.5
        else:
            ou_line = 2.5

        # 预计总进球数
        expected_goals = 2.8 * (1 + (home_prob - 0.5) * 0.35 + (away_prob - 0.5) * 0.35)
        expected_goals = max(1.5, min(4.5, expected_goals))

        # 判断大小球
        is_over = expected_goals > ou_line
        over_confidence = abs(expected_goals - ou_line) / ou_line
        over_confidence = min(0.85, 0.5 + over_confidence * 0.5)

        return {
            "line": ou_line,
            "expected_total": round(expected_goals, 1),
            "prediction": "大球" if is_over else "小球",
            "confidence": round(over_confidence, 4),
        }

    def _predict_basketball_spread(self, home_win_prob: float, match_data: Dict) -> float:
        """预测篮球让分"""
        # 胜率与让分的近似线性关系
        # 60% 胜率约等于让 3-4 分
        spread = (home_win_prob - 0.5) * 20  # -10 到 +10 分范围
        return spread

    def _predict_total_points(self, match_data: Dict) -> float:
        """预测篮球总得分（简化模型）"""
        # 基础总得分
        base_total = 210.0
        # 根据联赛调整
        league = match_data.get("league", "")
        if "NBA" in league.upper():
            base_total = 225.0
        elif "CBA" in league.upper():
            base_total = 195.0
        return base_total


# ============================================================
# 串关选择逻辑
# ============================================================
class ParlaySelector:
    """串关选择器"""

    def __init__(self, sport_type: str):
        self.sport_type = sport_type.lower()

    def select_parlay(
        self,
        predictions: List[Dict],
        min_confidence: float = 0.55,
        max_matches: int = 4,
        strategy: str = "balanced"
    ) -> Dict:
        """
        选择串关组合

        strategy:
            - "safe": 低风险，2-3场，高信心
            - "balanced": 平衡，3-4场，中高信心
            - "aggressive": 高风险，4-6场，中等信心
        """
        # 策略参数（v1.7 降低信心门槛，让更多比赛进入串关）
        strategies = {
            "safe": {"min_conf": 0.40, "min_games": 2, "max_games": 2},
            "balanced": {"min_conf": 0.35, "min_games": 2, "max_games": min(3, max_matches)},
            "aggressive": {"min_conf": 0.30, "min_games": 2, "max_games": min(4, max_matches)},
        }
        params = strategies.get(strategy, strategies["balanced"])

        # 过滤达到信心阈值的比赛
        filtered = [
            p for p in predictions
            if p.get("confidence", 0) >= params["min_conf"]
        ]

        # 按信心排序
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # 选择比赛
        selected_count = min(params["max_games"], len(filtered))
        selected_count = max(params["min_games"], min(selected_count, params["max_games"]))
        selected = filtered[:selected_count]

        # 计算串关综合胜率（v1.9 加入冷门相关性修正）
        combined_prob = 1.0
        avg_upset = 0.0
        upset_count = 0
        for p in selected:
            combined_prob *= p.get("confidence", 0.5)
            upset = p.get("upset_analysis", {}).get("upset_probability", 0.3)
            avg_upset += upset
            upset_count += 1

        if upset_count > 0:
            avg_upset /= upset_count

        # 冷门相关性修正：同一轮比赛冷门可能集中出现
        # 修正系数 = 0.95^(n-1)，n为比赛场数
        correlation_penalty = (0.95 ** (len(selected) - 1)) if len(selected) >= 2 else 1.0
        # 冷门平均概率越高，修正越强
        upset_penalty = 1.0 - (avg_upset * 0.15 * (len(selected) - 1))
        upset_penalty = max(0.70, min(1.0, upset_penalty))
        adjusted_combined_prob = combined_prob * correlation_penalty * upset_penalty

        # 估算奖金倍数（简化模型）
        # 实际赔率 ≈ 1 / 胜率 * 0.85（庄家抽水）
        total_odds = 1.0
        for p in selected:
            win_prob = p.get("confidence", 0.5)
            odds = 1.0 / max(win_prob, 0.01) * 0.85
            total_odds *= odds

        # 生成不同串关方式
        parlays = self._generate_parlay_variants(selected, strategy)

        return {
            "sport_type": self.sport_type,
            "strategy": strategy,
            "selected_matches": [
                {
                    "match_id": p["match_id"],
                    "home_team": p["home_team"],
                    "away_team": p["away_team"],
                    "recommendation": p["recommendation"],
                    "confidence": p["confidence"],
                    "league": p.get("league", ""),
                    "upset_probability": p.get("upset_analysis", {}).get("upset_probability", 0)
                }
                for p in selected
            ],
            "match_count": len(selected),
            "combined_probability": round(combined_prob, 4),
            "adjusted_combined_probability": round(adjusted_combined_prob, 4),  # v1.9 修正后
            "estimated_odds": round(total_odds, 2),
            "parlay_variants": parlays,
            "risk_level": self._assess_risk(adjusted_combined_prob, len(selected)),  # v1.9 使用修正后概率
            "upset_penalty": round(1.0 - upset_penalty, 4),  # v1.9 冷门折损率
            "average_upset_rate": round(avg_upset, 4),  # v1.9 平均冷门率
        }

    def _generate_parlay_variants(self, selected: List[Dict], strategy: str) -> List[Dict]:
        """生成不同串关方式"""
        n = len(selected)
        variants = []

        if n < 2:
            return variants

        # 单关（不串）
        variants.append({
            "type": "单关",
            "count": n,
            "description": f"{n}场单关，分散风险"
        })

        # 2串1
        if n >= 2:
            from itertools import combinations
            two_combs = len(list(combinations(range(n), 2)))
            variants.append({
                "type": "2串1",
                "count": two_combs,
                "description": f"全部{two_combs}组2串1组合"
            })

        # 3串1
        if n >= 3:
            three_combs = len(list(combinations(range(n), 3)))
            variants.append({
                "type": "3串1",
                "count": three_combs,
                "description": f"全部{three_combs}组3串1组合"
            })

        # N串1（全串）
        if n >= 2:
            variants.append({
                "type": f"{n}串1",
                "count": 1,
                "description": f"{n}场全串，高风险高回报"
            })

        # 推荐方案
        if strategy == "safe" and n >= 3:
            variants.append({
                "type": "3串4",
                "count": 4,
                "description": "3串4（含3个2串1+1个3串1），容错1场"
            })
        elif strategy == "balanced" and n >= 4:
            variants.append({
                "type": "4串11",
                "count": 11,
                "description": "4串11（含6个2串1+4个3串1+1个4串1），容错2场"
            })

        return variants

    def _assess_risk(self, combined_prob: float, match_count: int) -> str:
        """评估风险等级"""
        if combined_prob >= 0.5:
            return "低"
        elif combined_prob >= 0.3:
            return "中低"
        elif combined_prob >= 0.15:
            return "中"
        elif combined_prob >= 0.05:
            return "中高"
        else:
            return "高"


# ============================================================
# 主函数
# ============================================================
def generate_sports_prediction(sport_type: str, matches: List[Dict]) -> Dict:
    """
    生成竞彩预测结果（主函数）

    参数:
        sport_type: "football" 或 "basketball"
        matches: 比赛数据列表

    返回:
        包含单场预测和串关推荐的完整预测结果
    """
    sport_type = sport_type.lower()
    if sport_type not in ("football", "basketball"):
        return {
            "error": f"不支持的运动类型: {sport_type}",
            "version": VERSION,
            "timestamp": datetime.now().isoformat()
        }

    predictor = MatchPredictor(sport_type)
    parlay_selector = ParlaySelector(sport_type)

    # 单场预测
    single_predictions = []
    for match in matches:
        try:
            pred = predictor.predict(match)
            single_predictions.append(pred)
        except Exception as e:
            single_predictions.append({
                "match_id": match.get("match_id", "unknown"),
                "home_team": match.get("home_team", "未知"),
                "away_team": match.get("away_team", "未知"),
                "error": str(e)
            })

    # 串关推荐（多种策略）
    parlays = {}
    for strategy in ["safe", "balanced", "aggressive"]:
        parlays[strategy] = parlay_selector.select_parlay(
            single_predictions,
            strategy=strategy
        )

    # 总体统计
    total_matches = len(single_predictions)
    avg_confidence = (
        sum(p.get("confidence", 0) for p in single_predictions) / total_matches
        if total_matches > 0 else 0
    )
    # v1.9 新增冷门统计
    avg_upset = (
        sum(p.get("upset_analysis", {}).get("upset_probability", 0) for p in single_predictions) / total_matches
        if total_matches > 0 else 0
    )
    max_upset = max(
        (p.get("upset_analysis", {}).get("upset_probability", 0) for p in single_predictions),
        default=0
    )

    result = {
        "version": VERSION,
        "sport_type": sport_type,
        "timestamp": datetime.now().isoformat(),
        "total_matches": total_matches,
        "average_confidence": round(avg_confidence, 4),
        "average_upset_probability": round(avg_upset, 4),  # v1.9 新增
        "max_upset_probability": round(max_upset, 4),  # v1.9 新增
        "single_predictions": single_predictions,
        "parlay_recommendations": parlays,
        "weights": FOOTBALL_WEIGHTS if sport_type == "football" else BASKETBALL_WEIGHTS
    }

    return result


# ============================================================
# 便捷函数
# ============================================================
def predict_football(matches: List[Dict]) -> Dict:
    """便捷函数：足球预测"""
    return generate_sports_prediction("football", matches)


def predict_basketball(matches: List[Dict]) -> Dict:
    """便捷函数：篮球预测"""
    return generate_sports_prediction("basketball", matches)


# ============================================================
# 测试/示例
# ============================================================
def _demo():
    """演示预测引擎使用"""
    # 示例足球比赛
    football_matches = [
        {
            "match_id": "FB001",
            "league": "英超",
            "home_team": "曼城",
            "away_team": "利物浦",
            "home_rank": 1,
            "away_rank": 2,
            "total_teams": 20,
            "h2h": [
                {"result": "win"}, {"result": "draw"}, {"result": "win"},
                {"result": "lose"}, {"result": "win"}
            ],
            "home_form": ["W", "W", "W", "D", "W"],
            "away_form": ["W", "W", "D", "W", "L"],
            "home_home_record": {"wins": 8, "draws": 2, "losses": 0, "games": 10},
            "away_away_record": {"wins": 5, "draws": 3, "losses": 2, "games": 10},
            "home_injuries": [
                {"name": "德布劳内", "position": "MF", "importance": 9}
            ],
            "away_injuries": [],
            "home_last_match_days": 3,
            "away_last_match_days": 4,
            "home_upcoming": 1,
            "away_upcoming": 1,
            "weather": {
                "temperature": 22,
                "rainfall": 0,
                "wind_speed": 5,
                "condition": "sunny"
            }
        },
        {
            "match_id": "FB002",
            "league": "西甲",
            "home_team": "皇马",
            "away_team": "巴萨",
            "home_rank": 2,
            "away_rank": 3,
            "total_teams": 20,
            "h2h": [
                {"result": "win"}, {"result": "lose"}, {"result": "draw"},
                {"result": "win"}, {"result": "lose"}
            ],
            "home_form": ["W", "D", "W", "W", "W"],
            "away_form": ["W", "W", "W", "D", "W"],
            "home_home_record": {"wins": 7, "draws": 2, "losses": 1, "games": 10},
            "away_away_record": {"wins": 6, "draws": 2, "losses": 2, "games": 10},
            "home_injuries": [],
            "away_injuries": [
                {"name": "莱万多夫斯基", "position": "FW", "importance": 9},
                {"name": "佩德里", "position": "MF", "importance": 8}
            ],
            "home_last_match_days": 5,
            "away_last_match_days": 2,
            "home_upcoming": 0,
            "away_upcoming": 2,
            "weather": {
                "temperature": 28,
                "rainfall": 0,
                "wind_speed": 3,
                "condition": "sunny"
            }
        }
    ]

    # 示例篮球比赛
    basketball_matches = [
        {
            "match_id": "BB001",
            "league": "NBA",
            "home_team": "湖人",
            "away_team": "勇士",
            "home_rank": 5,
            "away_rank": 3,
            "total_teams": 30,
            "h2h": [
                {"result": "win"}, {"result": "lose"}, {"result": "win"}
            ],
            "home_form": ["W", "W", "L", "W", "W"],
            "away_form": ["W", "W", "W", "L", "W"],
            "home_home_record": {"wins": 15, "draws": 0, "losses": 8, "games": 23},
            "away_away_record": {"wins": 12, "draws": 0, "losses": 10, "games": 22},
            "home_injuries": [
                {"name": "詹姆斯", "position": "SF", "importance": 10}
            ],
            "away_injuries": [],
            "home_last_match_days": 2,
            "away_last_match_days": 1,
            "home_upcoming": 3,
            "away_upcoming": 2,
            "weather": {
                "temperature": 25,
                "rainfall": 0,
                "wind_speed": 0,
                "condition": "indoor"
            }
        }
    ]

    print(f"=== 竞彩预测算法引擎 {VERSION} ===")
    print()

    # 足球预测
    print("--- 足球预测 ---")
    fb_result = predict_football(football_matches)
    for pred in fb_result["single_predictions"]:
        print(f"  {pred['home_team']} vs {pred['away_team']}")
        print(f"    推荐: {pred['recommendation']} (信心: {pred['confidence']:.1%})")
        print(f"    比分预测: {pred['predicted_score']}")
        print(f"    概率: 主胜 {pred['probabilities']['home_win']:.1%}, "
              f"平 {pred['probabilities']['draw']:.1%}, "
              f"客胜 {pred['probabilities']['away_win']:.1%}")

    print(f"\n  串关推荐 (稳健型):")
    parlay = fb_result["parlay_recommendations"]["balanced"]
    print(f"    场次: {parlay['match_count']} 场")
    print(f"    综合胜率: {parlay['combined_probability']:.1%}")
    print(f"    估算赔率: {parlay['estimated_odds']:.2f}")
    print(f"    风险等级: {parlay['risk_level']}")

    print()

    # 篮球预测
    print("--- 篮球预测 ---")
    bb_result = predict_basketball(basketball_matches)
    for pred in bb_result["single_predictions"]:
        print(f"  {pred['home_team']} vs {pred['away_team']}")
        print(f"    推荐: {pred['recommendation']} (信心: {pred['confidence']:.1%})")
        print(f"    让分: {pred['spread']:+.1f}")
        print(f"    大小分: {pred['total_points']:.1f} ({pred['over_under']})")

    print()
    print("预测引擎运行完毕。")


if __name__ == "__main__":
    _demo()
