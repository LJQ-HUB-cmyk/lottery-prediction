#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日/每月总结生成器
读取预测数据和开奖结果，计算命中率和准确率，生成日/月度总结
"""

import json
import os
import sys
import glob
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SPORTS_PREDICTIONS_DIR = os.path.join(BASE_DIR, "sports_predictions")
SPORTS_RESULTS_DIR = os.path.join(BASE_DIR, "sports_results")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def load_json(file_path: str) -> Optional[Dict]:
    """加载 JSON 文件，不存在则返回 None"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [警告] 读取文件失败 {file_path}: {e}")
        return None


def save_json(file_path: str, data: Dict):
    """保存 JSON 文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_date_str(target_date: Optional[date] = None) -> str:
    """获取日期字符串 YYYY-MM-DD"""
    if target_date is None:
        target_date = date.today()
    return target_date.strftime("%Y-%m-%d")


def get_month_str(target_date: Optional[date] = None) -> str:
    """获取月份字符串 YYYY-MM"""
    if target_date is None:
        target_date = date.today()
    return target_date.strftime("%Y-%m")


# ============================================================
# 彩票命中率计算
# ============================================================
class LotteryHitCalculator:
    """彩票命中率计算器"""

    # 各彩种奖级定义
    PRIZE_LEVELS = {
        "SSQ": {  # 双色球
            "red_total": 33,
            "blue_total": 16,
            "red_pick": 6,
            "blue_pick": 1,
            "levels": [
                {"name": "一等奖", "red": 6, "blue": 1},
                {"name": "二等奖", "red": 6, "blue": 0},
                {"name": "三等奖", "red": 5, "blue": 1},
                {"name": "四等奖", "red": 5, "blue": 0},
                {"name": "四等奖", "red": 4, "blue": 1},
                {"name": "五等奖", "red": 4, "blue": 0},
                {"name": "五等奖", "red": 3, "blue": 1},
                {"name": "六等奖", "red": 2, "blue": 1},
                {"name": "六等奖", "red": 1, "blue": 1},
                {"name": "六等奖", "red": 0, "blue": 1},
            ]
        },
        "DLT": {  # 大乐透
            "red_total": 35,
            "blue_total": 12,
            "red_pick": 5,
            "blue_pick": 2,
            "levels": [
                {"name": "一等奖", "red": 5, "blue": 2},
                {"name": "二等奖", "red": 5, "blue": 1},
                {"name": "三等奖", "red": 5, "blue": 0},
                {"name": "四等奖", "red": 4, "blue": 2},
                {"name": "五等奖", "red": 4, "blue": 1},
                {"name": "六等奖", "red": 3, "blue": 2},
                {"name": "七等奖", "red": 4, "blue": 0},
                {"name": "八等奖", "red": 3, "blue": 1},
                {"name": "八等奖", "red": 2, "blue": 2},
                {"name": "九等奖", "red": 3, "blue": 0},
                {"name": "九等奖", "red": 2, "blue": 1},
                {"name": "九等奖", "red": 1, "blue": 2},
                {"name": "九等奖", "red": 0, "blue": 2},
            ]
        },
        "FC3D": {  # 福彩3D
            "digits": 3,
            "levels": [
                {"name": "直选", "type": "exact"},
                {"name": "组三", "type": "group3"},
                {"name": "组六", "type": "group6"},
            ]
        },
        "PL3": {  # 排列三
            "digits": 3,
            "levels": [
                {"name": "直选", "type": "exact"},
                {"name": "组三", "type": "group3"},
                {"name": "组六", "type": "group6"},
            ]
        },
        "PL5": {  # 排列五
            "digits": 5,
            "levels": [
                {"name": "一等奖", "type": "exact"},
            ]
        },
        "QLC": {  # 七乐彩
            "total": 30,
            "pick": 7,
            "levels": [
                {"name": "一等奖", "match": 7},
                {"name": "二等奖", "match": 6},
                {"name": "三等奖", "match": 5},
                {"name": "四等奖", "match": 4},
                {"name": "五等奖", "match": 3},
                {"name": "六等奖", "match": 2},
            ]
        },
    }

    @classmethod
    def calculate_hit(cls, lottery_type: str, prediction: Dict, result: Dict) -> Dict:
        """
        计算一组号码的命中情况

        返回: {"hit": bool, "level": str or None, "red_hit": int, "blue_hit": int}
        """
        lottery_type = lottery_type.upper()

        if lottery_type in ("SSQ", "DLT"):
            return cls._calc_lotto_hit(lottery_type, prediction, result)
        elif lottery_type in ("FC3D", "PL3"):
            return cls._calc_3d_hit(prediction, result)
        elif lottery_type == "PL5":
            return cls._calc_pl5_hit(prediction, result)
        elif lottery_type == "QLC":
            return cls._calc_qlc_hit(prediction, result)
        elif lottery_type == "QXC":
            return cls._calc_qxc_hit(prediction, result)
        else:
            # 通用：尝试匹配数字
            return cls._calc_generic_hit(prediction, result)

    @classmethod
    def _calc_lotto_hit(cls, lottery_type: str, prediction: Dict, result: Dict) -> Dict:
        """计算乐透型（双色球/大乐透）命中"""
        config = cls.PRIZE_LEVELS.get(lottery_type)
        if not config:
            return {"hit": False, "level": None, "red_hit": 0, "blue_hit": 0}

        # 提取红球和蓝球（兼容多种字段名）
        pred_red = set(prediction.get("red", prediction.get("red_balls", prediction.get("front", []))))
        pred_blue = set(prediction.get("blue", prediction.get("blue_balls", prediction.get("back", []))))
        res_red = set(result.get("red", result.get("red_balls", result.get("front", []))))
        res_blue = set(result.get("blue", result.get("blue_balls", result.get("back", []))))

        # 兼容数字和字符串
        pred_red = {str(x) for x in pred_red}
        pred_blue = {str(x) for x in pred_blue}
        res_red = {str(x) for x in res_red}
        res_blue = {str(x) for x in res_blue}

        red_hit = len(pred_red & res_red)
        blue_hit = len(pred_blue & res_blue)

        # 判断奖级
        hit_level = None
        for level in config["levels"]:
            if red_hit >= level["red"] and blue_hit >= level["blue"]:
                hit_level = level["name"]
                break

        return {
            "hit": hit_level is not None,
            "level": hit_level,
            "red_hit": red_hit,
            "blue_hit": blue_hit,
            "hit_desc": f"{red_hit}+{blue_hit}"
        }

    @classmethod
    def _calc_3d_hit(cls, prediction: Dict, result: Dict) -> Dict:
        """计算3D/排列三命中"""
        pred_digits = [str(d) for d in prediction.get("digits", prediction.get("numbers", []))]
        res_digits = [str(d) for d in result.get("digits", result.get("numbers", []))]

        if len(pred_digits) != 3 or len(res_digits) != 3:
            return {"hit": False, "level": None}

        # 直选
        exact = pred_digits == res_digits

        # 组选
        pred_sorted = sorted(pred_digits)
        res_sorted = sorted(res_digits)
        group_hit = pred_sorted == res_sorted

        # 判断组三组六
        unique_count = len(set(res_digits))

        hit_level = None
        if exact:
            hit_level = "直选"
        elif group_hit:
            if unique_count == 2:
                hit_level = "组三"
            elif unique_count == 3:
                hit_level = "组六"

        return {
            "hit": hit_level is not None,
            "level": hit_level,
            "exact": exact,
            "group_hit": group_hit
        }

    @classmethod
    def _calc_pl5_hit(cls, prediction: Dict, result: Dict) -> Dict:
        """计算排列五命中"""
        pred_digits = [str(d) for d in prediction.get("digits", prediction.get("numbers", []))]
        res_digits = [str(d) for d in result.get("digits", result.get("numbers", []))]

        exact = pred_digits == res_digits

        return {
            "hit": exact,
            "level": "一等奖" if exact else None,
            "exact": exact
        }

    @classmethod
    def _calc_qlc_hit(cls, prediction: Dict, result: Dict) -> Dict:
        """计算七乐彩命中"""
        pred_nums = {str(x) for x in prediction.get("numbers", prediction.get("red", prediction.get("red_balls", [])))}
        res_nums = {str(x) for x in result.get("numbers", result.get("red", result.get("red_balls", [])))}

        match_count = len(pred_nums & res_nums)

        config = cls.PRIZE_LEVELS.get("QLC", {})
        hit_level = None
        for level in config.get("levels", []):
            if match_count >= level["match"]:
                hit_level = level["name"]
                break

        return {
            "hit": hit_level is not None,
            "level": hit_level,
            "match_count": match_count
        }

    @classmethod
    def _calc_qxc_hit(cls, prediction: Dict, result: Dict) -> Dict:
        """计算七星彩命中（按位置匹配）"""
        pred_digits = [str(d) for d in prediction.get("digits", prediction.get("numbers", []))]
        res_digits = [str(d) for d in result.get("digits", result.get("numbers", []))]

        if len(pred_digits) != 7 or len(res_digits) != 7:
            return {"hit": False, "level": None, "position_hits": 0}

        # 按位置匹配
        position_hits = sum(1 for p, r in zip(pred_digits, res_digits) if p == r)

        # 七星彩奖级：按连续命中位数判断
        hit_level = None
        if position_hits == 7:
            hit_level = "一等奖"
        elif position_hits >= 6:
            hit_level = "二等奖"
        elif position_hits >= 5:
            hit_level = "三等奖"
        elif position_hits >= 4:
            hit_level = "四等奖"
        elif position_hits >= 3:
            hit_level = "五等奖"
        elif position_hits >= 2:
            hit_level = "六等奖"

        return {
            "hit": position_hits >= 2,
            "level": hit_level,
            "position_hits": position_hits,
            "hit_desc": f"{position_hits}/7位"
        }

    @classmethod
    def _calc_generic_hit(cls, prediction: Dict, result: Dict) -> Dict:
        """通用命中计算"""
        # 尝试多种字段名
        pred_nums = set()
        res_nums = set()

        for key in ["numbers", "red", "red_balls", "front", "digits", "balls"]:
            if key in prediction:
                pred_nums.update(str(x) for x in prediction[key])
            if key in result:
                res_nums.update(str(x) for x in result[key])

        match_count = len(pred_nums & res_nums)
        hit = match_count > 0

        return {
            "hit": hit,
            "level": None,
            "match_count": match_count
        }


# ============================================================
# 彩票总结生成
# ============================================================
def generate_lottery_summary(prediction_data: Dict, result_data: Dict) -> Dict:
    """
    生成彩票部分的总结

    返回各彩种的统计: {lottery_type: {group_count, groups_with_hit, hit_rate, max_hit_level}}
    """
    summary = {}

    if not prediction_data or not result_data:
        return summary

    # 遍历预测数据中的各彩种
    for lottery_type, pred_info in prediction_data.items():
        if not isinstance(pred_info, dict):
            continue

        # 获取该彩种的开奖结果（兼容 games 包装和直接格式）
        if 'games' in result_data:
            games_data = result_data['games']
        else:
            games_data = result_data
        result_info = games_data.get(lottery_type)
        if not result_info:
            continue

        # 提取预测组数
        groups = []
        if "groups" in pred_info and isinstance(pred_info["groups"], list):
            groups = pred_info["groups"]
        elif "predictions" in pred_info and isinstance(pred_info["predictions"], list):
            groups = pred_info["predictions"]
        elif "prediction" in pred_info and isinstance(pred_info["prediction"], dict):
            groups = [pred_info["prediction"]]
        elif "numbers" in pred_info:
            groups = [pred_info]

        if not groups:
            continue

        # 提取开奖号码
        draw_result = {}
        if "draw_result" in result_info:
            draw_result = result_info["draw_result"]
        elif "result" in result_info:
            draw_result = result_info["result"]
        else:
            draw_result = result_info

        # 计算每组命中情况
        group_count = len(groups)
        groups_with_hit = 0
        max_hit_level = None
        hit_details = []

        for group in groups:
            hit_info = LotteryHitCalculator.calculate_hit(
                lottery_type, group, draw_result
            )
            hit_details.append(hit_info)

            if hit_info.get("hit"):
                groups_with_hit += 1
                # 记录最高奖级
                level = hit_info.get("level")
                if level:
                    if max_hit_level is None or _is_higher_prize(level, max_hit_level, lottery_type):
                        max_hit_level = level

        # 如果没有中奖但有命中描述，记录最高命中描述
        if max_hit_level is None and hit_details:
            best = max(hit_details, key=lambda x: x.get("red_hit", 0) + x.get("blue_hit", 0) * 0.1)
            if best.get("hit_desc"):
                max_hit_level = best["hit_desc"]

        hit_rate = groups_with_hit / group_count if group_count > 0 else 0.0

        summary[lottery_type] = {
            "group_count": group_count,
            "groups_with_hit": groups_with_hit,
            "hit_rate": round(hit_rate, 4),
            "max_hit_level": max_hit_level,
            "hit_details": hit_details
        }

    return summary


def _is_higher_prize(level1: str, level2: str, lottery_type: str) -> bool:
    """判断 level1 是否比 level2 奖级更高"""
    config = LotteryHitCalculator.PRIZE_LEVELS.get(lottery_type.upper())
    if not config or "levels" not in config:
        # 简单比较：一等奖 > 二等奖 > ...
        order = ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖",
                 "七等奖", "八等奖", "九等奖", "直选", "组三", "组六"]
        idx1 = order.index(level1) if level1 in order else 99
        idx2 = order.index(level2) if level2 in order else 99
        return idx1 < idx2

    levels = [l["name"] for l in config["levels"]]
    # 去重保序
    seen = set()
    unique_levels = []
    for l in levels:
        if l not in seen:
            seen.add(l)
            unique_levels.append(l)

    idx1 = unique_levels.index(level1) if level1 in unique_levels else 99
    idx2 = unique_levels.index(level2) if level2 in unique_levels else 99
    return idx1 < idx2


# ============================================================
# 竞彩准确率计算
# ============================================================
def generate_sports_summary(prediction_data: Dict, result_data: Dict) -> Dict:
    """
    生成竞彩部分的总结

    返回各项目的统计: {football: {total, correct, accuracy}, basketball: {...}}
    """
    summary = {
        "football": {"total": 0, "correct": 0, "accuracy": 0.0, "details": []},
        "basketball": {"total": 0, "correct": 0, "accuracy": 0.0, "details": []},
    }

    if not prediction_data or not result_data:
        # 移除 details 字段
        return {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in summary.items()}

    # 处理足球
    fb_predictions = _extract_sports_predictions(prediction_data, "football")
    fb_results = _extract_sports_results(result_data, "football")
    fb_stats = _calc_sports_accuracy(fb_predictions, fb_results, "football")
    summary["football"] = fb_stats

    # 处理篮球
    bb_predictions = _extract_sports_predictions(prediction_data, "basketball")
    bb_results = _extract_sports_results(result_data, "basketball")
    bb_stats = _calc_sports_accuracy(bb_predictions, bb_results, "basketball")
    summary["basketball"] = bb_stats

    # 移除 details 字段，保留简洁版
    clean_summary = {}
    for sport, stats in summary.items():
        clean_stats = {k: v for k, v in stats.items() if k != "details"}
        clean_summary[sport] = clean_stats

    return clean_summary


def _extract_sports_predictions(data: Dict, sport: str) -> List[Dict]:
    """从预测数据中提取指定运动的单场预测列表"""
    predictions = []

    # 格式1: 直接按运动分类
    if sport in data and isinstance(data[sport], dict):
        sport_data = data[sport]
        if "single_predictions" in sport_data:
            predictions = sport_data["single_predictions"]
        elif "predictions" in sport_data:
            predictions = sport_data["predictions"]
        elif "matches" in sport_data:
            predictions = sport_data["matches"]

    # 格式2: 数据中直接是预测列表，按 sport_type 过滤
    if not predictions and "single_predictions" in data:
        predictions = [
            p for p in data["single_predictions"]
            if p.get("sport_type") == sport
        ]

    # 格式3: 直接就是列表
    if not predictions and isinstance(data, list):
        predictions = [p for p in data if p.get("sport_type") == sport]

    return predictions


def _extract_sports_results(data: Dict, sport: str) -> Dict:
    """从结果数据中提取指定运动的比赛结果字典（以 match_id 为 key）"""
    results = {}

    # 格式1: 按运动分类
    if sport in data and isinstance(data[sport], dict):
        sport_data = data[sport]
        if "results" in sport_data:
            for r in sport_data["results"]:
                mid = r.get("match_id", r.get("id", ""))
                if mid:
                    results[mid] = r
        elif "matches" in sport_data:
            for r in sport_data["matches"]:
                mid = r.get("match_id", r.get("id", ""))
                if mid:
                    results[mid] = r
        else:
            # 假设 key 就是 match_id
            for mid, r in sport_data.items():
                if isinstance(r, dict) and "result" in r:
                    results[mid] = r

    # 格式1b: 运动数据直接是列表（如 football: [{...}, ...]）
    if not results and sport in data and isinstance(data[sport], list):
        for r in data[sport]:
            mid = r.get("match_id", r.get("id", ""))
            if mid:
                results[mid] = r

    # 格式2: 整体是列表，按 sport_type 过滤
    if not results and isinstance(data, list):
        for r in data:
            if r.get("sport_type") == sport:
                mid = r.get("match_id", r.get("id", ""))
                if mid:
                    results[mid] = r

    # 格式3: 结果在 results 字段中
    if not results and "results" in data:
        res_list = data["results"]
        if isinstance(res_list, list):
            for r in res_list:
                if r.get("sport_type") == sport:
                    mid = r.get("match_id", r.get("id", ""))
                    if mid:
                        results[mid] = r

    return results


def _calc_sports_accuracy(
    predictions: List[Dict], results: Dict, sport: str
) -> Dict:
    """计算竞彩预测准确率"""
    total = 0
    correct = 0
    details = []

    for pred in predictions:
        match_id = pred.get("match_id", "")
        if not match_id:
            # 尝试通过队名匹配
            home = pred.get("home_team", "")
            away = pred.get("away_team", "")
            # 在结果中查找
            for rid, r in results.items():
                if (r.get("home_team") == home and r.get("away_team") == away):
                    match_id = rid
                    break

        if not match_id or match_id not in results:
            continue

        result = results[match_id]
        total += 1

        # 判断预测是否正确
        is_correct = _check_prediction_correct(pred, result, sport)
        if is_correct:
            correct += 1

        details.append({
            "match_id": match_id,
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "prediction": pred.get("recommendation", ""),
            "actual": result.get("result", result.get("outcome", "")),
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "details": details
    }


def _check_prediction_correct(pred: Dict, result: Dict, sport: str) -> bool:
    """检查单场预测是否正确"""
    recommendation = pred.get("recommendation", "")
    actual_result = result.get("result", result.get("outcome", ""))

    # 标准化结果
    actual = _normalize_result(actual_result, result)

    # 直接匹配
    if recommendation == actual:
        return True

    # 足球胜平负判断
    if sport == "football":
        home_score = result.get("home_score", result.get("home_goals"))
        away_score = result.get("away_score", result.get("away_goals"))

        if home_score is not None and away_score is not None:
            try:
                hs = int(home_score)
                as_ = int(away_score)
                if hs > as_:
                    actual_result_full = "主胜"
                elif hs == as_:
                    actual_result_full = "平局"
                else:
                    actual_result_full = "客胜"
                return recommendation == actual_result_full
            except (ValueError, TypeError):
                pass

    # 篮球胜负判断
    if sport == "basketball":
        home_score = result.get("home_score", result.get("home_points"))
        away_score = result.get("away_score", result.get("away_points"))

        if home_score is not None and away_score is not None:
            try:
                hs = int(home_score)
                as_ = int(away_score)
                if hs > as_:
                    actual_result_full = "主胜"
                else:
                    actual_result_full = "客胜"
                return recommendation == actual_result_full
            except (ValueError, TypeError):
                pass

    return False


def _normalize_result(result_str: str, result_data: Dict) -> str:
    """标准化结果字符串"""
    result_str = str(result_str).strip()

    mapping = {
        "主胜": "主胜", "home_win": "主胜", "home": "主胜", "胜": "主胜",
        "客胜": "客胜", "away_win": "客胜", "away": "客胜", "负": "客胜",
        "平局": "平局", "draw": "平局", "平": "平局", "d": "平局",
    }

    return mapping.get(result_str.lower(), result_str)


# ============================================================
# 每日总结生成
# ============================================================
def generate_daily_summary(target_date: Optional[date] = None) -> Dict:
    """
    生成每日总结

    返回总结字典，同时写入文件
    """
    date_str = get_date_str(target_date)
    print(f"生成 {date_str} 每日总结...")

    # 读取各类数据
    pred_file = os.path.join(PREDICTIONS_DIR, f"{date_str}.json")
    result_file = os.path.join(RESULTS_DIR, f"{date_str}.json")
    sports_pred_file = os.path.join(SPORTS_PREDICTIONS_DIR, f"{date_str}.json")
    sports_result_file = os.path.join(SPORTS_RESULTS_DIR, f"{date_str}.json")

    prediction_data = load_json(pred_file)
    result_data = load_json(result_file)
    sports_prediction_data = load_json(sports_pred_file)
    sports_result_data = load_json(sports_result_file)

    print(f"  彩票预测数据: {'已加载' if prediction_data else '未找到'}")
    print(f"  彩票开奖结果: {'已加载' if result_data else '未找到'}")
    print(f"  竞彩预测数据: {'已加载' if sports_prediction_data else '未找到'}")
    print(f"  竞彩比赛结果: {'已加载' if sports_result_data else '未找到'}")

    # 计算彩票命中率
    lottery_summary = generate_lottery_summary(prediction_data, result_data)

    # 计算竞彩准确率
    sports_summary = generate_sports_summary(sports_prediction_data, sports_result_data)

    # 组装总结
    summary = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "lottery": lottery_summary,
        "sports": sports_summary
    }

    # 保存
    output_file = os.path.join(SUMMARY_DIR, f"daily_{date_str}.json")
    save_json(output_file, summary)
    print(f"  每日总结已保存: {output_file}")

    return summary


# ============================================================
# 每月总结生成
# ============================================================
def generate_monthly_summary(target_date: Optional[date] = None) -> Dict:
    """
    生成月度总结（聚合该月所有日总结）

    返回总结字典，同时写入文件
    """
    month_str = get_month_str(target_date)
    print(f"生成 {month_str} 月度总结...")

    # 查找该月所有日总结文件
    pattern = os.path.join(SUMMARY_DIR, f"daily_{month_str}-*.json")
    daily_files = sorted(glob.glob(pattern))

    if not daily_files:
        print(f"  未找到 {month_str} 的日总结文件，尝试先生成当日总结...")
        # 尝试生成当日总结
        today = date.today()
        if today.strftime("%Y-%m") == month_str:
            generate_daily_summary(today)
            daily_files = sorted(glob.glob(pattern))

    print(f"  找到 {len(daily_files)} 个日总结文件")

    # 聚合统计
    lottery_stats = defaultdict(lambda: {
        "total_groups": 0,
        "total_hit_groups": 0,
        "total_days": 0,
        "hit_days": 0,
        "max_hit_level": None,
        "daily_hit_rates": []
    })

    sports_stats = defaultdict(lambda: {
        "total_predictions": 0,
        "total_correct": 0,
        "total_days": 0,
        "daily_accuracies": []
    })

    for daily_file in daily_files:
        daily_data = load_json(daily_file)
        if not daily_data:
            continue

        # 彩票统计
        for ltype, lstats in daily_data.get("lottery", {}).items():
            s = lottery_stats[ltype]
            s["total_groups"] += lstats.get("group_count", 0)
            s["total_hit_groups"] += lstats.get("groups_with_hit", 0)
            s["total_days"] += 1
            if lstats.get("groups_with_hit", 0) > 0:
                s["hit_days"] += 1
            s["daily_hit_rates"].append(lstats.get("hit_rate", 0))

            # 最高奖级
            current_max = lstats.get("max_hit_level")
            if current_max:
                if s["max_hit_level"] is None:
                    s["max_hit_level"] = current_max
                else:
                    if _is_higher_prize(current_max, s["max_hit_level"], ltype):
                        s["max_hit_level"] = current_max

        # 竞彩统计
        for sport, sstats in daily_data.get("sports", {}).items():
            s = sports_stats[sport]
            s["total_predictions"] += sstats.get("total", 0)
            s["total_correct"] += sstats.get("correct", 0)
            if sstats.get("total", 0) > 0:
                s["total_days"] += 1
                s["daily_accuracies"].append(sstats.get("accuracy", 0))

    # 格式化输出
    lottery_monthly = {}
    for ltype, s in lottery_stats.items():
        overall_hit_rate = (
            s["total_hit_groups"] / s["total_groups"]
            if s["total_groups"] > 0 else 0.0
        )
        avg_daily_rate = (
            sum(s["daily_hit_rates"]) / len(s["daily_hit_rates"])
            if s["daily_hit_rates"] else 0.0
        )
        hit_day_rate = s["hit_days"] / s["total_days"] if s["total_days"] > 0 else 0.0

        lottery_monthly[ltype] = {
            "total_days": s["total_days"],
            "total_groups": s["total_groups"],
            "total_hit_groups": s["total_hit_groups"],
            "overall_hit_rate": round(overall_hit_rate, 4),
            "avg_daily_hit_rate": round(avg_daily_rate, 4),
            "hit_day_rate": round(hit_day_rate, 4),
            "hit_days": s["hit_days"],
            "max_hit_level": s["max_hit_level"]
        }

    sports_monthly = {}
    for sport, s in sports_stats.items():
        overall_accuracy = (
            s["total_correct"] / s["total_predictions"]
            if s["total_predictions"] > 0 else 0.0
        )
        avg_daily_acc = (
            sum(s["daily_accuracies"]) / len(s["daily_accuracies"])
            if s["daily_accuracies"] else 0.0
        )

        sports_monthly[sport] = {
            "total_days": s["total_days"],
            "total_predictions": s["total_predictions"],
            "total_correct": s["total_correct"],
            "overall_accuracy": round(overall_accuracy, 4),
            "avg_daily_accuracy": round(avg_daily_acc, 4)
        }

    summary = {
        "month": month_str,
        "generated_at": datetime.now().isoformat(),
        "days_covered": len(daily_files),
        "lottery": lottery_monthly,
        "sports": sports_monthly
    }

    # 保存
    output_file = os.path.join(SUMMARY_DIR, f"monthly_{month_str}.json")
    save_json(output_file, summary)
    print(f"  月度总结已保存: {output_file}")

    return summary


# ============================================================
# 主入口
# ============================================================
def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="每日/每月总结生成器")
    parser.add_argument(
        "--date", "-d",
        help="指定日期 (YYYY-MM-DD)，默认为今天",
        default=None
    )
    parser.add_argument(
        "--month", "-m",
        help="指定月份 (YYYY-MM)，生成月度总结",
        default=None
    )
    parser.add_argument(
        "--type", "-t",
        choices=["daily", "monthly", "both"],
        default="both",
        help="生成类型: daily(每日), monthly(每月), both(都生成)"
    )

    args = parser.parse_args()

    # 解析日期
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"错误: 日期格式不正确 {args.date}，请使用 YYYY-MM-DD")
            sys.exit(1)

    target_month = None
    if args.month:
        try:
            target_month = datetime.strptime(args.month, "%Y-%m").date()
        except ValueError:
            print(f"错误: 月份格式不正确 {args.month}，请使用 YYYY-MM")
            sys.exit(1)

    ensure_dir(SUMMARY_DIR)

    # 生成每日总结
    if args.type in ("daily", "both"):
        try:
            generate_daily_summary(target_date)
        except Exception as e:
            print(f"生成每日总结时出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成月度总结
    if args.type in ("monthly", "both"):
        try:
            generate_monthly_summary(target_month or target_date)
        except Exception as e:
            print(f"生成月度总结时出错: {e}")
            import traceback
            traceback.print_exc()

    print("总结生成完成。")


if __name__ == "__main__":
    main()
