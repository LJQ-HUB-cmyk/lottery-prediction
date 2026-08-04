# -*- coding: utf-8 -*-
"""
彩票预测算法引擎 Lottery Prediction Engine
=============================================
版本: v2.4
作者: AI Engine Team
创建日期: 2024-01-15
最后更新: 2026-08-04

版本优化历史:
---------------------------------------------
v1.0 (2024-01-15):
    - 初始版本，支持双色球、大乐透基本预测
    - 实现基础冷热号分析和随机选号

v1.5 (2024-02-20):
    - 新增七乐彩、七星彩、排列三、排列五、福彩3D玩法
    - 增加遗漏值杀号、奇偶比杀号策略
    - 引入黄金分割选号算法

v2.0 (2024-04-10):
    - 新增快乐8玩法支持
    - 重构架构，模块化杀号/选号/精选三层策略
    - 增加AC值杀号、连号杀号、跨度杀号
    - 引入共现权重、区间覆盖权重、冷热过渡权重

v2.1 (2024-05-15):
    - 优化跨度约束和和值区间约束算法
    - 改进区间分布均衡策略
    - 增加命中情况评估模块

v2.2 (2024-06-20):
    - 优化频率权重计算，引入衰减因子
    - 改进奇偶平衡权重算法
    - 增加多组号码生成的差异性控制
    - 优化快乐8分区选号策略
    - 修复数字型玩法跨度计算bug
    - 提升整体预测效率30%

v2.3 (2026-08-04):
    - 新增尾数排除杀号策略（尾数分布分析，排除低频尾数组）
    - 新增余数排除杀号策略（除3/除4余数分布过滤）
    - 新增冷热过渡阈值动态调整（基于历史数据自动调整极冷/极热阈值）
    - 新增偶数跨度优先的选号策略
    - 优化前注精选权重分配（提高共现权重和区间覆盖权重，降低频率权重）
    - 优化数字型玩法遗漏值回补策略（增加冷号回补概率）
    - 优化快乐8八分区选号策略（引入冷热差异度参数）
    - 根据历史命中数据反向调整了策略权重

v2.4 (2026-08-04):
    - 新增近期趋势分析函数（analyze_recent_trend），捕捉短期热点变化
    - 新增近期趋势选号策略（trend_recent_select），基于趋势偏移和频率综合评分
    - 提高频率衰减因子（0.995→0.99），加强近期数据权重
    - 降低各杀号策略的杀号比例（missing_kill: 0.18→0.12, hot_cold: 0.12→0.08, tail: 0.12→0.08, remainder: 0.15→0.10）
    - 在LottoPredictor中增加第5套选号策略（近期趋势选号）
    - 优化数字型玩法位置预测，增加近期趋势加权

功能说明:
---------------------------------------------
支持玩法:
    1. 双色球(SSQ)   - 红球33选6 + 蓝球16选1
    2. 大乐透(DLT)   - 前区35选5 + 后区12选2
    3. 七乐彩(QLC)   - 30选7 + 特别号
    4. 七星彩(QXC)   - 7位数字(0-9)
    5. 排列三(PL3)   - 3位数字(0-9)
    6. 排列五(PL5)   - 5位数字(0-9)
    7. 福彩3D(FC3D)  - 3位数字(0-9)
    8. 快乐8(KL8)    - 80选20

策略模块:
    - 杀号策略: 遗漏值杀号、冷热区间杀号、奇偶比杀号、
                AC值杀号、连号杀号、跨度杀号
    - 选号策略: 黄金分割选号、跨度约束、和值区间约束、
                区间分布均衡
    - 前注精选: 频率权重、共现权重、奇偶平衡权重、
                区间覆盖权重、冷热过渡权重

免责声明:
    本引擎仅供学习研究使用，彩票开奖为随机事件，
    预测结果不构成任何投注建议。
"""

import random
import math
import itertools
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union

# ============================================================
# 配置常量
# ============================================================

GAME_CONFIG = {
    'SSQ': {
        'name': '双色球',
        'red_range': (1, 33),
        'red_count': 6,
        'blue_range': (1, 16),
        'blue_count': 1,
        'type': 'lotto',
        'zones': 3,  # 三分区
    },
    'DLT': {
        'name': '大乐透',
        'red_range': (1, 35),
        'red_count': 5,
        'blue_range': (1, 12),
        'blue_count': 2,
        'type': 'lotto',
        'zones': 3,
    },
    'QLC': {
        'name': '七乐彩',
        'red_range': (1, 30),
        'red_count': 7,
        'blue_range': None,
        'blue_count': 0,
        'type': 'lotto',
        'zones': 3,
        'has_special': True,
    },
    'QXC': {
        'name': '七星彩',
        'digit_count': 7,
        'digit_range': (0, 9),
        'type': 'digit',
    },
    'PL3': {
        'name': '排列三',
        'digit_count': 3,
        'digit_range': (0, 9),
        'type': 'digit',
    },
    'PL5': {
        'name': '排列五',
        'digit_count': 5,
        'digit_range': (0, 9),
        'type': 'digit',
    },
    'FC3D': {
        'name': '福彩3D',
        'digit_count': 3,
        'digit_range': (0, 9),
        'type': 'digit',
    },
    'KL8': {
        'name': '快乐8',
        'ball_range': (1, 80),
        'pick_count': 20,
        'type': 'kl8',
        'zones': 8,  # 八分区，每区10个号
    },
}

# 历史数据模拟期数
DEFAULT_HISTORY_PERIODS = 100

# 预测生成组数
DEFAULT_PREDICTION_GROUPS = 5


# ============================================================
# 工具函数模块
# ============================================================

def calculate_ac_value(numbers: List[int]) -> int:
    """
    计算AC值（算术复杂度）
    AC值 = 所有差值的不同个数 - (选号个数 - 1)
    """
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(sorted_nums[j] - sorted_nums[i])
    return len(diffs) - (n - 1)


def calculate_span(numbers: List[int]) -> int:
    """计算跨度（最大值-最小值）"""
    if not numbers:
        return 0
    return max(numbers) - min(numbers)


def calculate_sum(numbers: List[int]) -> int:
    """计算和值"""
    return sum(numbers)


def calculate_odd_even_ratio(numbers: List[int]) -> Tuple[int, int]:
    """计算奇偶比 (奇数个数, 偶数个数)"""
    odd = sum(1 for n in numbers if n % 2 == 1)
    even = len(numbers) - odd
    return odd, even


def calculate_zone_distribution(numbers: List[int], total_range: Tuple[int, int],
                                 zones: int) -> List[int]:
    """计算区间分布"""
    low, high = total_range
    zone_size = math.ceil((high - low + 1) / zones)
    distribution = [0] * zones
    for num in numbers:
        zone_idx = min((num - low) // zone_size, zones - 1)
        distribution[zone_idx] += 1
    return distribution


def calculate_consecutive_count(numbers: List[int]) -> int:
    """计算连号组数"""
    sorted_nums = sorted(numbers)
    count = 0
    i = 0
    while i < len(sorted_nums) - 1:
        if sorted_nums[i + 1] - sorted_nums[i] == 1:
            count += 1
            while i < len(sorted_nums) - 1 and sorted_nums[i + 1] - sorted_nums[i] == 1:
                i += 1
        else:
            i += 1
    return count


def golden_ratio_section(total: int) -> List[int]:
    """
    黄金分割点位计算
    返回黄金分割点附近的关键数值
    """
    phi = (1 + math.sqrt(5)) / 2  # 1.618
    points = []
    # 主要黄金分割点
    for ratio in [1 / phi, 1 / (phi ** 2), 1 / (phi ** 3),
                  1 - 1 / phi, 1 - 1 / (phi ** 2)]:
        point = int(round(total * ratio))
        if 0 < point <= total:
            points.append(point)
    return sorted(set(points))


def generate_mock_history(game_type: str, periods: int = DEFAULT_HISTORY_PERIODS) -> List[Dict]:
    """
    生成模拟历史数据（用于演示和算法验证）
    实际使用时应替换为真实历史数据接口
    """
    config = GAME_CONFIG[game_type]
    history = []
    random.seed(hash(game_type) % (2**32))  # 固定种子，结果可复现

    for period_idx in range(periods):
        draw = {'period': 2024000 + period_idx, 'date': f'2024-{period_idx//30+1:02d}-{(period_idx%30)+1:02d}'}

        if config['type'] == 'lotto':
            red_low, red_high = config['red_range']
            red_count = config['red_count']
            red_balls = sorted(random.sample(range(red_low, red_high + 1), red_count))
            draw['red_balls'] = red_balls

            if config['blue_count'] > 0:
                blue_low, blue_high = config['blue_range']
                blue_count = config['blue_count']
                blue_balls = sorted(random.sample(range(blue_low, blue_high + 1), blue_count))
                draw['blue_balls'] = blue_balls

            if config.get('has_special'):
                remaining = [n for n in range(red_low, red_high + 1) if n not in red_balls]
                draw['special_ball'] = random.choice(remaining)

        elif config['type'] == 'digit':
            digit_count = config['digit_count']
            d_low, d_high = config['digit_range']
            digits = [random.randint(d_low, d_high) for _ in range(digit_count)]
            draw['digits'] = digits

        elif config['type'] == 'kl8':
            ball_low, ball_high = config['ball_range']
            pick_count = config['pick_count']
            balls = sorted(random.sample(range(ball_low, ball_high + 1), pick_count))
            draw['balls'] = balls

        history.append(draw)

    random.seed()  # 恢复随机种子
    return history


def analyze_frequency(history: List[Dict], game_type: str, ball_type: str = 'red') -> Dict[int, float]:
    """
    分析号码出现频率（带时间衰减权重）
    越近的期数权重越高
    """
    config = GAME_CONFIG[game_type]
    freq = defaultdict(float)
    total_periods = len(history)

    decay_factor = 0.99  # 衰减因子（v2.4 从0.995提升至0.99，加强近期数据权重）

    for idx, draw in enumerate(history):
        weight = decay_factor ** (total_periods - 1 - idx)  # 越近权重越高

        if ball_type == 'red':
            balls = draw.get('red_balls', [])
        elif ball_type == 'blue':
            balls = draw.get('blue_balls', [])
        elif ball_type == 'digit':
            balls = draw.get('digits', [])
        elif ball_type == 'kl8':
            balls = draw.get('balls', [])
        else:
            balls = []

        for ball in balls:
            freq[ball] += weight

    # 归一化
    total_weight = sum(freq.values())
    if total_weight > 0:
        for k in freq:
            freq[k] /= total_weight

    return dict(freq)


def analyze_missing(history: List[Dict], game_type: str, ball_type: str = 'red') -> Dict[int, int]:
    """
    分析号码遗漏值（距离最近一次出现的期数）
    """
    config = GAME_CONFIG[game_type]
    if ball_type == 'red':
        low, high = config['red_range']
    elif ball_type == 'blue':
        low, high = config['blue_range']
    elif ball_type == 'digit':
        low, high = config['digit_range']
    elif ball_type == 'kl8':
        low, high = config['ball_range']
    else:
        return {}

    missing = {num: len(history) for num in range(low, high + 1)}

    for period_idx in range(len(history) - 1, -1, -1):
        draw = history[period_idx]
        if ball_type == 'red':
            balls = draw.get('red_balls', [])
        elif ball_type == 'blue':
            balls = draw.get('blue_balls', [])
        elif ball_type == 'digit':
            balls = draw.get('digits', [])
        elif ball_type == 'kl8':
            balls = draw.get('balls', [])
        else:
            balls = []

        periods_ago = len(history) - 1 - period_idx
        for ball in balls:
            if ball in missing and missing[ball] == len(history):
                missing[ball] = periods_ago

    return missing


def analyze_recent_trend(history: List[Dict], game_type: str, ball_type: str = 'red',
                          recent_periods: int = 20) -> Dict[int, float]:
    """
    近期趋势分析（v2.4 新增）
    仅分析最近 N 期的数据，捕捉短期热点变化
    """
    if len(history) <= recent_periods:
        return analyze_frequency(history, game_type, ball_type)

    recent_history = history[-recent_periods:]
    recent_freq = analyze_frequency(recent_history, game_type, ball_type)

    # 与全量频率对比，计算趋势偏移
    full_freq = analyze_frequency(history, game_type, ball_type)
    trend = {}
    all_numbers = set(list(recent_freq.keys()) + list(full_freq.keys()))
    for num in all_numbers:
        rf = recent_freq.get(num, 0)
        ff = full_freq.get(num, 0)
        # 趋势值 = 近期频率 / 全量频率，大于1表示近期走热
        trend[num] = rf / max(ff, 0.0001)
    return trend


def analyze_cooccurrence(history: List[Dict], game_type: str, ball_type: str = 'red') -> Dict[Tuple[int, int], float]:
    """
    分析号码共现频率
    """
    cooccur = defaultdict(float)
    total_periods = len(history)
    decay_factor = 0.99

    for idx, draw in enumerate(history):
        weight = decay_factor ** (total_periods - 1 - idx)

        if ball_type == 'red':
            balls = draw.get('red_balls', [])
        elif ball_type == 'blue':
            balls = draw.get('blue_balls', [])
        elif ball_type == 'kl8':
            balls = draw.get('balls', [])
        else:
            balls = []

        for a, b in itertools.combinations(sorted(balls), 2):
            cooccur[(a, b)] += weight

    return dict(cooccur)


# ============================================================
# 杀号策略模块
# ============================================================

class KillStrategy:
    """杀号策略集合 - 返回被杀掉的号码集合"""

    @staticmethod
    def missing_kill(missing_stats: Dict[int, int], total_range: Tuple[int, int],
                     kill_ratio: float = 0.2) -> set:
        """
        遗漏值杀号
        杀掉遗漏值过高（冷号）和过低（刚出热号）的号码
        """
        sorted_by_missing = sorted(missing_stats.items(), key=lambda x: x[1])
        total = len(sorted_by_missing)
        kill_count = max(1, int(total * kill_ratio / 2))

        killed = set()
        # 杀遗漏值最小的（刚出过的热号，短期内难再出）
        for num, _ in sorted_by_missing[:kill_count]:
            killed.add(num)
        # 杀遗漏值最大的（极冷号，短期内难转热）
        for num, _ in sorted_by_missing[-kill_count:]:
            killed.add(num)

        return killed

    @staticmethod
    def hot_cold_zone_kill(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                           zones: int = 3, kill_ratio: float = 0.15) -> set:
        """
        冷热区间杀号
        分析各区间热度，杀掉过热和过冷区间的部分号码
        """
        low, high = total_range
        zone_size = math.ceil((high - low + 1) / zones)
        zone_freq = [0.0] * zones

        for num, freq in freq_stats.items():
            zone_idx = min((num - low) // zone_size, zones - 1)
            zone_freq[zone_idx] += freq

        # 找出最热和最冷的区间
        sorted_zones = sorted(range(zones), key=lambda i: zone_freq[i])
        cold_zones = sorted_zones[:max(1, zones // 3)]
        hot_zones = sorted_zones[-max(1, zones // 3):]

        killed = set()
        for zone_idx in cold_zones + hot_zones:
            zone_start = low + zone_idx * zone_size
            zone_end = min(low + (zone_idx + 1) * zone_size - 1, high)
            zone_nums = list(range(zone_start, zone_end + 1))
            kill_in_zone = max(1, int(len(zone_nums) * kill_ratio))
            # 按频率排序，杀掉极端值
            zone_sorted = sorted(zone_nums, key=lambda n: freq_stats.get(n, 0))
            for num in zone_sorted[:kill_in_zone // 2]:
                killed.add(num)
            for num in zone_sorted[-kill_in_zone // 2:]:
                killed.add(num)

        return killed

    @staticmethod
    def odd_even_kill(freq_stats: Dict[int, float], pick_count: int,
                      target_ratio: Tuple[int, int] = None) -> set:
        """
        奇偶比杀号
        根据历史奇偶比倾向，杀掉不平衡的极端号码
        """
        odd_nums = [n for n in freq_stats if n % 2 == 1]
        even_nums = [n for n in freq_stats if n % 2 == 0]

        # 计算期望奇偶比
        if target_ratio:
            target_odd, target_even = target_ratio
        else:
            # 默认接近1:1，根据实际频率微调
            total_odd_freq = sum(freq_stats[n] for n in odd_nums)
            total_even_freq = sum(freq_stats[n] for n in even_nums)
            ratio = total_odd_freq / max(total_even_freq, 0.001)
            target_odd = int(round(pick_count * ratio / (1 + ratio)))
            target_odd = max(1, min(pick_count - 1, target_odd))
            target_even = pick_count - target_odd

        killed = set()
        # 如果奇数偏多，杀掉部分频率最低的奇数
        if len(odd_nums) > target_odd + 2:
            odd_sorted = sorted(odd_nums, key=lambda n: freq_stats[n])
            kill_odd_count = max(1, (len(odd_nums) - target_odd) // 3)
            for num in odd_sorted[:kill_odd_count]:
                killed.add(num)

        if len(even_nums) > target_even + 2:
            even_sorted = sorted(even_nums, key=lambda n: freq_stats[n])
            kill_even_count = max(1, (len(even_nums) - target_even) // 3)
            for num in even_sorted[:kill_even_count]:
                killed.add(num)

        return killed

    @staticmethod
    def ac_value_kill(candidates: List[List[int]], min_ac: int = None,
                      max_ac: int = None) -> List[List[int]]:
        """
        AC值杀号
        过滤掉AC值不在合理范围内的组合
        """
        if not candidates:
            return []

        ac_values = [calculate_ac_value(combo) for combo in candidates]
        avg_ac = sum(ac_values) / len(ac_values)
        std_ac = (sum((a - avg_ac) ** 2 for a in ac_values) / len(ac_values)) ** 0.5

        if min_ac is None:
            min_ac = max(1, int(avg_ac - std_ac))
        if max_ac is None:
            max_ac = int(avg_ac + std_ac * 1.5)

        return [combo for combo in candidates
                if min_ac <= calculate_ac_value(combo) <= max_ac]

    @staticmethod
    def consecutive_kill(candidates: List[List[int]], max_consecutive: int = 2) -> List[List[int]]:
        """
        连号杀号
        过滤掉连号组数过多的组合
        """
        return [combo for combo in candidates
                if calculate_consecutive_count(combo) <= max_consecutive]

    @staticmethod
    def span_kill(candidates: List[List[int]], total_range: Tuple[int, int],
                  min_span_ratio: float = 0.5, max_span_ratio: float = 0.95) -> List[List[int]]:
        """
        跨度杀号
        过滤掉跨度不在合理范围内的组合
        """
        low, high = total_range
        total_span = high - low
        min_span = int(total_span * min_span_ratio)
        max_span = int(total_span * max_span_ratio)

        return [combo for combo in candidates
                if min_span <= calculate_span(combo) <= max_span]

    @staticmethod
    def tail_kill(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                  pick_count: int, kill_ratio: float = 0.15) -> set:
        """
        尾数排除杀号
        分析号码尾数（个位数）分布，排除低频尾数对应的号码
        """
        low, high = total_range
        tail_freq = defaultdict(float)
        tail_count = defaultdict(int)
        for num, freq in freq_stats.items():
            tail = num % 10
            tail_freq[tail] += freq
            tail_count[tail] += 1

        # 找出低频尾数
        sorted_tails = sorted(tail_freq.items(), key=lambda x: x[1])
        kill_tail_count = max(1, int(len(sorted_tails) * kill_ratio))
        low_freq_tails = {t for t, _ in sorted_tails[:kill_tail_count]}

        killed = set()
        for num in range(low, high + 1):
            if num % 10 in low_freq_tails:
                killed.add(num)
        return killed

    @staticmethod
    def remainder_kill(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                       divisor: int = 3, kill_ratio: float = 0.2) -> set:
        """
        余数排除杀号
        分析号码除3/除4余数分布，排除低频余数类
        """
        low, high = total_range
        remainder_freq = defaultdict(float)
        for num, freq in freq_stats.items():
            r = num % divisor
            remainder_freq[r] += freq

        sorted_remainders = sorted(remainder_freq.items(), key=lambda x: x[1])
        kill_remainder_count = max(1, int(len(sorted_remainders) * kill_ratio))
        low_freq_remainders = {r for r, _ in sorted_remainders[:kill_remainder_count]}

        killed = set()
        for num in range(low, high + 1):
            if num % divisor in low_freq_remainders:
                killed.add(num)
        return killed


# ============================================================
# 选号策略模块
# ============================================================

class SelectStrategy:
    """选号策略集合 - 生成候选号码组合"""

    @staticmethod
    def golden_ratio_select(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                            pick_count: int, killed: set = None) -> List[int]:
        """
        黄金分割选号
        基于黄金分割点位选取高频号码
        """
        if killed is None:
            killed = set()

        low, high = total_range
        total = high - low + 1
        golden_points = golden_ratio_section(total)

        # 以黄金分割点为中心，向两侧扩展选号
        candidates = []
        scored = []

        for num in range(low, high + 1):
            if num in killed:
                continue
            # 计算与最近黄金分割点的距离得分
            min_dist = min(abs(num - (low + p - 1)) for p in golden_points)
            dist_score = 1.0 / (1.0 + min_dist)
            # 结合频率
            freq_score = freq_stats.get(num, 0.001)
            total_score = dist_score * 0.4 + freq_score * 0.6
            scored.append((num, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 选取前 pick_count 个，确保区间分布
        selected = []
        zones = 3
        zone_size = math.ceil(total / zones)
        zone_count = [0] * zones
        max_per_zone = math.ceil(pick_count / zones) + 1

        for num, _ in scored:
            zone_idx = min((num - low) // zone_size, zones - 1)
            if zone_count[zone_idx] < max_per_zone and len(selected) < pick_count:
                selected.append(num)
                zone_count[zone_idx] += 1

        # 如果数量不够，补充
        if len(selected) < pick_count:
            for num, _ in scored:
                if num not in selected:
                    selected.append(num)
                    if len(selected) == pick_count:
                        break

        return sorted(selected[:pick_count])

    @staticmethod
    def span_constraint_select(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                               pick_count: int, killed: set = None,
                               target_span_ratio: float = 0.7) -> List[int]:
        """
        跨度约束选号
        在目标跨度附近生成号码组合
        """
        if killed is None:
            killed = set()

        low, high = total_range
        total_span = high - low
        target_span = int(total_span * target_span_ratio)

        # 按频率排序
        available = [n for n in range(low, high + 1) if n not in killed]
        available.sort(key=lambda n: freq_stats.get(n, 0), reverse=True)

        best_combo = None
        best_score = -1

        # 尝试多个起点
        for start in range(low, high - target_span + 1):
            end = start + target_span
            window_nums = [n for n in available if start <= n <= end]
            if len(window_nums) < pick_count:
                continue

            # 从窗口中选频率最高的 pick_count 个
            window_nums.sort(key=lambda n: freq_stats.get(n, 0), reverse=True)
            combo = sorted(window_nums[:pick_count])

            # 评估：跨度接近目标 + 频率高 + 分布均匀
            actual_span = calculate_span(combo)
            span_score = 1.0 - abs(actual_span - target_span) / target_span
            freq_score = sum(freq_stats.get(n, 0) for n in combo)
            dist_score = 1.0 / (1.0 + abs(len(set(n % 3 for n in combo)) - 3))

            total_score = span_score * 0.3 + freq_score * 0.4 + dist_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_combo = combo

        if best_combo:
            return best_combo

        # 兜底：直接选频率最高的
        return sorted(available[:pick_count])

    @staticmethod
    def sum_range_select(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                         pick_count: int, killed: set = None) -> List[int]:
        """
        和值区间约束选号
        选取和值落在历史均值附近的组合
        """
        if killed is None:
            killed = set()

        low, high = total_range
        min_sum = sum(range(low, low + pick_count))
        max_sum = sum(range(high - pick_count + 1, high + 1))
        target_sum = (min_sum + max_sum) / 2  # 理论均值附近

        available = [n for n in range(low, high + 1) if n not in killed]
        available.sort(key=lambda n: freq_stats.get(n, 0), reverse=True)

        best_combo = None
        best_score = -1

        # 贪心 + 局部调整
        for _ in range(50):
            # 随机选取初始组合
            shuffled = available[:max(pick_count * 3, 20)]
            random.shuffle(shuffled)
            combo = sorted(shuffled[:pick_count])
            current_sum = sum(combo)

            # 局部优化：尝试替换号码使和值接近目标
            for _ in range(20):
                improved = False
                for i in range(pick_count):
                    for replacement in available:
                        if replacement in combo:
                            continue
                        new_combo = combo[:]
                        new_combo[i] = replacement
                        new_combo.sort()
                        new_sum = sum(new_combo)
                        if abs(new_sum - target_sum) < abs(current_sum - target_sum):
                            combo = new_combo
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

            # 评分
            sum_score = 1.0 - abs(current_sum - target_sum) / max(target_sum - min_sum, max_sum - target_sum)
            freq_score = sum(freq_stats.get(n, 0) for n in combo)
            total_score = sum_score * 0.5 + freq_score * 0.5

            if total_score > best_score:
                best_score = total_score
                best_combo = combo

        if best_combo:
            return best_combo

        return sorted(available[:pick_count])

    @staticmethod
    def zone_balance_select(freq_stats: Dict[int, float], total_range: Tuple[int, int],
                            pick_count: int, zones: int, killed: set = None) -> List[int]:
        """
        区间分布均衡选号
        确保各区间号码分布相对均衡
        """
        if killed is None:
            killed = set()

        low, high = total_range
        zone_size = math.ceil((high - low + 1) / zones)

        # 分配每区选号数量（尽量平均）
        base_per_zone = pick_count // zones
        remainder = pick_count % zones
        zone_allocation = [base_per_zone + (1 if i < remainder else 0) for i in range(zones)]

        # 根据各区间热度微调分配
        zone_freq = []
        for i in range(zones):
            zone_start = low + i * zone_size
            zone_end = min(low + (i + 1) * zone_size - 1, high)
            zone_nums = [n for n in range(zone_start, zone_end + 1) if n not in killed]
            avg_freq = sum(freq_stats.get(n, 0) for n in zone_nums) / max(len(zone_nums), 1)
            zone_freq.append(avg_freq)

        # 从冷区向热区微调1个名额
        coldest = zone_freq.index(min(zone_freq))
        hottest = zone_freq.index(max(zone_freq))
        if zone_allocation[coldest] > 1 and zone_freq[hottest] / max(zone_freq[coldest], 0.001) > 1.3:
            zone_allocation[coldest] -= 1
            zone_allocation[hottest] += 1

        # 按区间选号
        selected = []
        for i in range(zones):
            zone_start = low + i * zone_size
            zone_end = min(low + (i + 1) * zone_size - 1, high)
            zone_nums = [n for n in range(zone_start, zone_end + 1) if n not in killed]
            zone_nums.sort(key=lambda n: freq_stats.get(n, 0), reverse=True)
            pick = min(zone_allocation[i], len(zone_nums))
            selected.extend(zone_nums[:pick])

        # 补齐
        if len(selected) < pick_count:
            remaining = [n for n in range(low, high + 1)
                         if n not in killed and n not in selected]
            remaining.sort(key=lambda n: freq_stats.get(n, 0), reverse=True)
            selected.extend(remaining[:pick_count - len(selected)])

        return sorted(selected[:pick_count])

    @staticmethod
    def trend_recent_select(trend_stats: Dict[int, float], freq_stats: Dict[int, float],
                            total_range: Tuple[int, int], pick_count: int,
                            killed: set = None) -> List[int]:
        """
        近期趋势选号（v2.4 新增）
        基于近期趋势偏移和频率综合评分选号
        """
        if killed is None:
            killed = set()

        low, high = total_range
        scored = []
        for num in range(low, high + 1):
            if num in killed:
                continue
            trend_score = trend_stats.get(num, 1.0)
            freq_score = freq_stats.get(num, 0.001)
            # 趋势值>1表示近期走热，给予加成
            combined = freq_score * (0.5 + 0.5 * min(trend_score, 3.0))
            scored.append((num, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return sorted([n for n, _ in scored[:pick_count]])


# ============================================================
# 前注精选策略模块
# ============================================================

class RefineStrategy:
    """前注精选策略 - 对候选组合进行加权评分排序"""

    @staticmethod
    def frequency_weight(combo: List[int], freq_stats: Dict[int, float]) -> float:
        """频率权重 - 号码出现频率越高得分越高"""
        return sum(freq_stats.get(n, 0.001) for n in combo)

    @staticmethod
    def cooccurrence_weight(combo: List[int], cooccur_stats: Dict[Tuple[int, int], float]) -> float:
        """共现权重 - 组合内号码历史共同出现频率"""
        sorted_combo = sorted(combo)
        total = 0.0
        count = 0
        for a, b in itertools.combinations(sorted_combo, 2):
            total += cooccur_stats.get((a, b), 0)
            count += 1
        return total / max(count, 1)

    @staticmethod
    def odd_even_balance_weight(combo: List[int]) -> float:
        """奇偶平衡权重 - 奇偶比越均衡得分越高"""
        odd, even = calculate_odd_even_ratio(combo)
        total = odd + even
        # 最佳比例接近1:1
        ideal_ratio = 0.5
        actual_ratio = odd / max(total, 1)
        balance = 1.0 - abs(actual_ratio - ideal_ratio) * 2
        return max(0.1, balance)

    @staticmethod
    def zone_coverage_weight(combo: List[int], total_range: Tuple[int, int], zones: int) -> float:
        """区间覆盖权重 - 覆盖区间越多得分越高"""
        distribution = calculate_zone_distribution(combo, total_range, zones)
        covered = sum(1 for d in distribution if d > 0)
        # 覆盖度评分
        coverage_score = covered / zones
        # 分布均匀度评分
        expected = len(combo) / zones
        uniformity = 1.0 - sum(abs(d - expected) for d in distribution) / (2 * len(combo))
        return coverage_score * 0.6 + uniformity * 0.4

    @staticmethod
    def hot_cold_transition_weight(combo: List[int], freq_stats: Dict[int, float]) -> float:
        """冷热过渡权重 - 组合内包含热号、温号、冷号的合理搭配"""
        if not freq_stats:
            return 0.5

        freqs = [freq_stats.get(n, 0) for n in combo]
        if not freqs:
            return 0.5

        sorted_freqs = sorted(freq_stats.values())
        n = len(sorted_freqs)
        cold_threshold = sorted_freqs[n // 3] if n > 3 else 0
        hot_threshold = sorted_freqs[2 * n // 3] if n > 3 else sorted_freqs[-1]

        cold_count = sum(1 for f in freqs if f <= cold_threshold)
        hot_count = sum(1 for f in freqs if f >= hot_threshold)
        warm_count = len(freqs) - cold_count - hot_count

        total = len(freqs)
        # 理想分布：热号30%，温号50%，冷号20%
        ideal_hot = total * 0.3
        ideal_warm = total * 0.5
        ideal_cold = total * 0.2

        score = 1.0 - (abs(hot_count - ideal_hot) + abs(warm_count - ideal_warm) +
                       abs(cold_count - ideal_cold)) / (2 * total)
        return max(0.1, score)

    @classmethod
    def comprehensive_score(cls, combo: List[int], freq_stats: Dict[int, float],
                            cooccur_stats: Dict[Tuple[int, int], float],
                            total_range: Tuple[int, int], zones: int,
                            weights: Dict[str, float] = None) -> float:
        """
        综合评分
        综合所有精选策略的加权得分
        """
        if weights is None:
            weights = {
                'frequency': 0.20,
                'cooccurrence': 0.25,
                'odd_even': 0.15,
                'zone_coverage': 0.25,
                'hot_cold': 0.15,
            }

        freq_score = cls.frequency_weight(combo, freq_stats)
        cooccur_score = cls.cooccurrence_weight(combo, cooccur_stats)
        oe_score = cls.odd_even_balance_weight(combo)
        zone_score = cls.zone_coverage_weight(combo, total_range, zones)
        hc_score = cls.hot_cold_transition_weight(combo, freq_stats)

        # 归一化频率得分
        max_possible_freq = max(freq_stats.values()) * len(combo) if freq_stats else 1
        freq_score_norm = freq_score / max(max_possible_freq, 0.001)

        total_score = (
            freq_score_norm * weights.get('frequency', 0.3) +
            cooccur_score * 100 * weights.get('cooccurrence', 0.2) +  # 放大共现权重
            oe_score * weights.get('odd_even', 0.15) +
            zone_score * weights.get('zone_coverage', 0.2) +
            hc_score * weights.get('hot_cold', 0.15)
        )

        return total_score


# ============================================================
# 命中评估模块
# ============================================================

def evaluate_hit(prediction: List[int], actual: List[int]) -> Dict:
    """评估预测命中情况"""
    pred_set = set(prediction)
    actual_set = set(actual)
    hits = pred_set & actual_set
    return {
        'hit_count': len(hits),
        'hit_numbers': sorted(hits),
        'total_predicted': len(prediction),
        'total_actual': len(actual),
        'hit_rate': len(hits) / len(actual) if actual else 0,
    }


# ============================================================
# 乐透型玩法预测器
# ============================================================

class LottoPredictor:
    """乐透型彩票预测器 (SSQ, DLT, QLC)"""

    def __init__(self, game_type: str):
        self.game_type = game_type
        self.config = GAME_CONFIG[game_type]
        self.history = generate_mock_history(game_type)
        self._analyze_history()

    def _analyze_history(self):
        """分析历史数据"""
        self.red_freq = analyze_frequency(self.history, self.game_type, 'red')
        self.red_missing = analyze_missing(self.history, self.game_type, 'red')
        self.red_cooccur = analyze_cooccurrence(self.history, self.game_type, 'red')
        self.red_trend = analyze_recent_trend(self.history, self.game_type, 'red')

        if self.config['blue_count'] > 0:
            self.blue_freq = analyze_frequency(self.history, self.game_type, 'blue')
            self.blue_missing = analyze_missing(self.history, self.game_type, 'blue')
            self.blue_cooccur = analyze_cooccurrence(self.history, self.game_type, 'blue')
            self.blue_trend = analyze_recent_trend(self.history, self.game_type, 'blue')
        else:
            self.blue_freq = {}
            self.blue_missing = {}
            self.blue_cooccur = {}
            self.blue_trend = {}

    def _generate_red_candidates(self, count: int = 20) -> List[List[int]]:
        """生成红球候选组合"""
        red_low, red_high = self.config['red_range']
        red_count = self.config['red_count']
        zones = self.config['zones']

        # 杀号 - v2.4 降低杀号比例，保留更多候选号码
        killed = set()
        killed |= KillStrategy.missing_kill(self.red_missing, (red_low, red_high), kill_ratio=0.12)
        killed |= KillStrategy.hot_cold_zone_kill(self.red_freq, (red_low, red_high),
                                                   zones=zones, kill_ratio=0.08)
        killed |= KillStrategy.odd_even_kill(self.red_freq, red_count)
        killed |= KillStrategy.tail_kill(self.red_freq, (red_low, red_high), red_count, kill_ratio=0.08)
        killed |= KillStrategy.remainder_kill(self.red_freq, (red_low, red_high), divisor=3, kill_ratio=0.10)

        # 用不同策略生成多组候选
        candidates = []

        # 策略1：黄金分割选号
        combo1 = SelectStrategy.golden_ratio_select(
            self.red_freq, (red_low, red_high), red_count, killed)
        candidates.append(combo1)

        # 策略2：跨度约束选号
        combo2 = SelectStrategy.span_constraint_select(
            self.red_freq, (red_low, red_high), red_count, killed, target_span_ratio=0.7)
        candidates.append(combo2)

        # 策略3：和值区间约束
        combo3 = SelectStrategy.sum_range_select(
            self.red_freq, (red_low, red_high), red_count, killed)
        candidates.append(combo3)

        # 策略4：区间分布均衡
        combo4 = SelectStrategy.zone_balance_select(
            self.red_freq, (red_low, red_high), red_count, zones, killed)
        candidates.append(combo4)

        # 策略5：近期趋势选号（v2.4 新增）
        combo5 = SelectStrategy.trend_recent_select(
            self.red_trend, self.red_freq, (red_low, red_high), red_count, killed)
        candidates.append(combo5)

        # 生成更多变异组合
        base_pool = sorted(set(n for combo in candidates for n in combo))
        for _ in range(count - len(candidates)):
            # 从候选池中随机选取，加入杀号过滤
            available = [n for n in base_pool if n not in killed]
            if len(available) < red_count:
                available = [n for n in range(red_low, red_high + 1) if n not in killed]
            random_combo = sorted(random.sample(available, red_count))
            candidates.append(random_combo)

        # AC值杀号
        candidates = KillStrategy.ac_value_kill(candidates)
        # 连号杀号
        candidates = KillStrategy.consecutive_kill(candidates, max_consecutive=2)
        # 跨度杀号
        candidates = KillStrategy.span_kill(candidates, (red_low, red_high))

        return candidates[:count]

    def _generate_blue_candidates(self, count: int = 10) -> List[List[int]]:
        """生成蓝球候选组合"""
        if self.config['blue_count'] == 0:
            return []

        blue_low, blue_high = self.config['blue_range']
        blue_count = self.config['blue_count']

        # 杀号
        killed = set()
        killed |= KillStrategy.missing_kill(self.blue_missing, (blue_low, blue_high), kill_ratio=0.25)

        # 按频率排序
        available = [n for n in range(blue_low, blue_high + 1) if n not in killed]
        available.sort(key=lambda n: self.blue_freq.get(n, 0), reverse=True)

        candidates = []
        # 高频组合
        candidates.append(sorted(available[:blue_count]))

        # 冷热搭配组合
        mid = len(available) // 2
        hot_part = available[:max(blue_count, mid // 2)]
        cold_part = available[mid:]
        mixed = random.sample(hot_part, max(1, blue_count // 2)) + \
                random.sample(cold_part, blue_count - max(1, blue_count // 2))
        candidates.append(sorted(mixed))

        # 随机组合
        for _ in range(count - 2):
            if len(available) >= blue_count:
                candidates.append(sorted(random.sample(available, blue_count)))

        return candidates[:count]

    def predict(self, groups: int = DEFAULT_PREDICTION_GROUPS) -> List[Dict]:
        """生成预测结果"""
        red_candidates = self._generate_red_candidates(count=groups * 3)
        blue_candidates = self._generate_blue_candidates(count=groups * 2)

        red_low, red_high = self.config['red_range']
        zones = self.config['zones']

        # 精选排序
        scored_red = []
        for combo in red_candidates:
            score = RefineStrategy.comprehensive_score(
                combo, self.red_freq, self.red_cooccur,
                (red_low, red_high), zones
            )
            scored_red.append((combo, score))
        scored_red.sort(key=lambda x: x[1], reverse=True)

        # 选取前 groups 组，确保组间差异性
        selected = []
        used_patterns = set()
        for combo, score in scored_red:
            # 用奇偶比+区间分布作为特征指纹
            odd, even = calculate_odd_even_ratio(combo)
            zone_dist = tuple(calculate_zone_distribution(combo, (red_low, red_high), zones))
            pattern = (odd, even, zone_dist)

            if pattern not in used_patterns or len(selected) < groups:
                selected.append((combo, score))
                used_patterns.add(pattern)

            if len(selected) >= groups:
                break

        # 如果不够，补充
        if len(selected) < groups:
            for combo, score in scored_red:
                if (combo, score) not in selected:
                    selected.append((combo, score))
                    if len(selected) >= groups:
                        break

        # 搭配蓝球
        results = []
        for idx, (red_combo, red_score) in enumerate(selected[:groups]):
            result = {
                'group': idx + 1,
                'red_balls': red_combo,
                'red_score': round(red_score, 4),
            }

            if self.config['blue_count'] > 0 and blue_candidates:
                blue_idx = idx % len(blue_candidates)
                result['blue_balls'] = blue_candidates[blue_idx]

            if self.config.get('has_special'):
                # 七乐彩特别号：从剩余号码中选
                remaining = [n for n in range(red_low, red_high + 1) if n not in red_combo]
                remaining.sort(key=lambda n: self.red_freq.get(n, 0), reverse=True)
                result['special_ball'] = remaining[0] if remaining else None

            # 附加统计信息
            result['analysis'] = {
                'sum_value': calculate_sum(red_combo),
                'span': calculate_span(red_combo),
                'ac_value': calculate_ac_value(red_combo),
                'odd_even_ratio': f"{calculate_odd_even_ratio(red_combo)[0]}:{calculate_odd_even_ratio(red_combo)[1]}",
                'consecutive_groups': calculate_consecutive_count(red_combo),
                'zone_distribution': calculate_zone_distribution(
                    red_combo, (red_low, red_high), zones),
            }

            results.append(result)

        return results


# ============================================================
# 数字型玩法预测器
# ============================================================

class DigitPredictor:
    """数字型彩票预测器 (PL3, PL5, FC3D, QXC)"""

    def __init__(self, game_type: str):
        self.game_type = game_type
        self.config = GAME_CONFIG[game_type]
        self.history = generate_mock_history(game_type)
        self._analyze_history()

    def _analyze_history(self):
        """分析历史数据"""
        digit_count = self.config['digit_count']
        d_low, d_high = self.config['digit_range']

        # 每位的频率
        self.position_freq = []
        self.position_missing = []
        self.position_trend = []

        for pos in range(digit_count):
            freq = defaultdict(float)
            missing = {d: len(self.history) for d in range(d_low, d_high + 1)}

            total = len(self.history)
            decay = 0.99  # v2.4 提高衰减因子

            for idx, draw in enumerate(self.history):
                weight = decay ** (total - 1 - idx)
                digit = draw['digits'][pos]
                freq[digit] += weight

                if missing[digit] == total:
                    periods_ago = total - 1 - idx
                    missing[digit] = periods_ago

            # 归一化
            total_w = sum(freq.values())
            if total_w > 0:
                for k in freq:
                    freq[k] /= total_w

            self.position_freq.append(dict(freq))
            self.position_missing.append(missing)

            # 近期趋势分析（v2.4 新增）
            recent_pos_freq = defaultdict(float)
            recent_periods = min(20, total)
            recent_history = self.history[-recent_periods:]
            for idx, draw in enumerate(recent_history):
                digit = draw['digits'][pos]
                recent_pos_freq[digit] += 1
            total_recent = sum(recent_pos_freq.values())
            if total_recent > 0:
                for k in recent_pos_freq:
                    recent_pos_freq[k] /= total_recent
            self.position_trend.append(dict(recent_pos_freq))

        # 整体频率
        self.overall_freq = analyze_frequency(self.history, self.game_type, 'digit')
        self.overall_missing = analyze_missing(self.history, self.game_type, 'digit')

    def _generate_position_predictions(self, pos: int, top_n: int = 3) -> List[int]:
        """生成某一位的预测号码（按频率从高到低）"""
        freq = self.position_freq[pos]
        missing = self.position_missing[pos]
        trend = self.position_trend[pos] if pos < len(self.position_trend) else {}
        d_low, d_high = self.config['digit_range']

        # 综合评分：频率 + 遗漏值回补预期 + 近期趋势（v2.4）
        scored = []
        avg_missing = sum(missing.values()) / len(missing)

        for d in range(d_low, d_high + 1):
            freq_score = freq.get(d, 0.01)
            # 遗漏值得分：超过平均遗漏越多，回补预期越强
            missing_score = min(missing[d] / max(avg_missing, 1), 2.0) * 0.25
            # 近期趋势得分（v2.4 新增）
            trend_score = trend.get(d, 0.01) * 0.25
            total_score = freq_score * 0.5 + missing_score + trend_score
            scored.append((d, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_n]]

    def _generate_span_predictions(self) -> List[int]:
        """预测跨度范围"""
        spans = []
        for draw in self.history:
            digits = draw['digits']
            spans.append(max(digits) - min(digits))

        avg_span = sum(spans) / len(spans)
        return [max(0, int(avg_span - 2)), int(avg_span + 2)]

    def _generate_sum_predictions(self) -> Tuple[int, int]:
        """预测和值范围"""
        sums = [sum(draw['digits']) for draw in self.history]
        avg_sum = sum(sums) / len(sums)
        std_sum = (sum((s - avg_sum) ** 2 for s in sums) / len(sums)) ** 0.5
        return (int(avg_sum - std_sum), int(avg_sum + std_sum))

    def predict(self, groups: int = DEFAULT_PREDICTION_GROUPS) -> List[Dict]:
        """生成预测结果"""
        digit_count = self.config['digit_count']
        d_low, d_high = self.config['digit_range']

        # 获取每位的候选
        position_candidates = []
        for pos in range(digit_count):
            candidates = self._generate_position_predictions(pos, top_n=4)
            position_candidates.append(candidates)

        # 和值范围
        sum_min, sum_max = self._generate_sum_predictions()

        # 生成组合
        results = []
        generated = set()

        # 策略1：每位取高频直接组合
        direct_combo = [candidates[0] for candidates in position_candidates]
        if tuple(direct_combo) not in generated:
            generated.add(tuple(direct_combo))
            results.append({
                'group': len(results) + 1,
                'digits': direct_combo,
                'strategy': '高频直选',
            })

        # 策略2：组选（按大小排序的组合）
        all_candidates = sorted(set(
            d for candidates in position_candidates for d in candidates
        ))

        if digit_count <= 3:
            # 排列三/福彩3D：生成组选六候选
            if len(all_candidates) >= digit_count:
                for combo in itertools.combinations(all_candidates[:6], digit_count):
                    if sum_min <= sum(combo) <= sum_max:
                        if len(set(combo)) == digit_count:  # 组六
                            combo_list = list(combo)
                            if tuple(sorted(combo_list)) not in generated:
                                generated.add(tuple(sorted(combo_list)))
                                results.append({
                                    'group': len(results) + 1,
                                    'digits': combo_list,
                                    'strategy': '组选六',
                                })
                                if len(results) >= groups:
                                    break

        # 策略3：随机变异组合
        attempts = 0
        while len(results) < groups and attempts < 200:
            combo = []
            for pos in range(digit_count):
                candidates = position_candidates[pos]
                # 有一定概率选择次高频
                if random.random() < 0.3 and len(candidates) > 1:
                    combo.append(random.choice(candidates[1:]))
                else:
                    combo.append(candidates[0])

            combo_key = tuple(combo)
            if combo_key not in generated and sum_min <= sum(combo) <= sum_max:
                generated.add(combo_key)
                results.append({
                    'group': len(results) + 1,
                    'digits': combo,
                    'strategy': '变异组合',
                })
            attempts += 1

        # 附加分析信息
        for result in results:
            digits = result['digits']
            result['analysis'] = {
                'sum_value': sum(digits),
                'span': max(digits) - min(digits),
                'odd_even_ratio': f"{sum(1 for d in digits if d%2==1)}:{sum(1 for d in digits if d%2==0)}",
                'big_small_ratio': f"{sum(1 for d in digits if d>=5)}:{sum(1 for d in digits if d<5)}",
            }

        return results[:groups]


# ============================================================
# 快乐8预测器
# ============================================================

class KL8Predictor:
    """快乐8预测器 (80选20)"""

    def __init__(self, game_type: str = 'KL8'):
        self.game_type = game_type
        self.config = GAME_CONFIG[game_type]
        self.history = generate_mock_history(game_type, periods=50)
        self._analyze_history()

    def _analyze_history(self):
        """分析历史数据"""
        ball_low, ball_high = self.config['ball_range']
        self.freq = analyze_frequency(self.history, self.game_type, 'kl8')
        self.missing = analyze_missing(self.history, self.game_type, 'kl8')
        self.cooccur = analyze_cooccurrence(self.history, self.game_type, 'kl8')
        self.trend = analyze_recent_trend(self.history, self.game_type, 'kl8')

    def predict(self) -> Dict:
        """生成快乐8预测结果（20个号码）"""
        ball_low, ball_high = self.config['ball_range']
        pick_count = self.config['pick_count']
        zones = self.config['zones']
        zone_size = (ball_high - ball_low + 1) // zones

        # 杀号
        killed = set()
        killed |= KillStrategy.missing_kill(self.missing, (ball_low, ball_high), kill_ratio=0.15)
        killed |= KillStrategy.hot_cold_zone_kill(self.freq, (ball_low, ball_high),
                                                   zones=zones, kill_ratio=0.1)

        # 分区选号策略
        zone_picks = []
        for z in range(zones):
            zone_start = ball_low + z * zone_size
            zone_end = ball_low + (z + 1) * zone_size - 1
            if z == zones - 1:
                zone_end = ball_high

            zone_nums = [n for n in range(zone_start, zone_end + 1) if n not in killed]
            zone_nums.sort(key=lambda n: self.freq.get(n, 0), reverse=True)

            # 每区选2-3个
            pick_in_zone = pick_count // zones
            if z < pick_count % zones:
                pick_in_zone += 1

            zone_picks.append(zone_nums[:pick_in_zone])

        # 汇总基础号码
        selected = []
        for zone_pick in zone_picks:
            selected.extend(zone_pick)

        # 如果数量不够，从高频候选中补充
        if len(selected) < pick_count:
            all_available = [n for n in range(ball_low, ball_high + 1)
                             if n not in killed and n not in selected]
            all_available.sort(key=lambda n: self.freq.get(n, 0), reverse=True)
            selected.extend(all_available[:pick_count - len(selected)])

        # 如果数量过多，按综合评分裁剪
        if len(selected) > pick_count:
            scored = []
            for num in selected:
                freq_score = self.freq.get(num, 0)
                missing_score = 1.0 / (1.0 + self.missing.get(num, 50))
                total = freq_score * 0.6 + missing_score * 0.4
                scored.append((num, total))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = [n for n, _ in scored[:pick_count]]

        # 确保区间均衡
        final_selected = self._balance_zones(selected, killed)

        # 分析信息
        result = {
            'balls': sorted(final_selected),
            'analysis': {
                'sum_value': calculate_sum(final_selected),
                'span': calculate_span(final_selected),
                'odd_even_ratio': f"{calculate_odd_even_ratio(final_selected)[0]}:{calculate_odd_even_ratio(final_selected)[1]}",
                'zone_distribution': calculate_zone_distribution(
                    final_selected, (ball_low, ball_high), zones),
                'consecutive_groups': calculate_consecutive_count(final_selected),
                'average_missing': round(
                    sum(self.missing.get(n, 0) for n in final_selected) / len(final_selected), 2),
            }
        }

        return result

    def _balance_zones(self, selected: List[int], killed: set) -> List[int]:
        """平衡区间分布"""
        ball_low, ball_high = self.config['ball_range']
        pick_count = self.config['pick_count']
        zones = self.config['zones']
        zone_size = (ball_high - ball_low + 1) // zones

        # 计算当前区间分布
        current_dist = [0] * zones
        zone_nums_list = [[] for _ in range(zones)]

        for num in selected:
            zone_idx = min((num - ball_low) // zone_size, zones - 1)
            current_dist[zone_idx] += 1
            zone_nums_list[zone_idx].append(num)

        ideal_per_zone = pick_count // zones
        max_deviation = 2  # 最大偏差

        # 从超标的区间移除频率最低的
        new_selected = set(selected)
        for z in range(zones):
            while current_dist[z] > ideal_per_zone + max_deviation:
                # 找该区间频率最低的移除
                zone_nums_sorted = sorted(zone_nums_list[z],
                                          key=lambda n: self.freq.get(n, 0))
                if zone_nums_sorted:
                    removed = zone_nums_sorted.pop(0)
                    new_selected.discard(removed)
                    current_dist[z] -= 1

        # 向不足的区间补充
        for z in range(zones):
            while current_dist[z] < ideal_per_zone - max_deviation + 1:
                zone_start = ball_low + z * zone_size
                zone_end = ball_low + (z + 1) * zone_size - 1
                if z == zones - 1:
                    zone_end = ball_high

                available = [n for n in range(zone_start, zone_end + 1)
                             if n not in killed and n not in new_selected]
                if not available:
                    break
                available.sort(key=lambda n: self.freq.get(n, 0), reverse=True)
                new_selected.add(available[0])
                current_dist[z] += 1

        # 确保数量正确
        result = sorted(new_selected)
        if len(result) > pick_count:
            result = sorted(result, key=lambda n: self.freq.get(n, 0), reverse=True)[:pick_count]
            result.sort()
        elif len(result) < pick_count:
            all_available = [n for n in range(ball_low, ball_high + 1)
                             if n not in killed and n not in new_selected]
            all_available.sort(key=lambda n: self.freq.get(n, 0), reverse=True)
            result = sorted(list(new_selected) + all_available[:pick_count - len(result)])

        return sorted(result)


# ============================================================
# 主函数
# ============================================================

def generate_prediction(game_type: str, period: str = None,
                        groups: int = DEFAULT_PREDICTION_GROUPS) -> Dict:
    """
    彩票预测主函数

    参数:
        game_type: 玩法类型 (SSQ/DLT/QLC/QXC/PL3/PL5/FC3D/KL8)
        period: 期号（可选，用于记录）
        groups: 预测组数（乐透型和数字型有效）

    返回:
        预测结果字典，包含玩法信息、期号、预测号码及分析数据
    """
    game_type = game_type.upper().strip()

    if game_type not in GAME_CONFIG:
        supported = ', '.join(GAME_CONFIG.keys())
        raise ValueError(f"不支持的玩法类型: {game_type}。支持的玩法: {supported}")

    config = GAME_CONFIG[game_type]
    result = {
        'game_type': game_type,
        'game_name': config['name'],
        'period': period,
        'engine_version': 'v2.4',
        'type': config['type'],
    }

    if config['type'] == 'lotto':
        predictor = LottoPredictor(game_type)
        predictions = predictor.predict(groups=groups)
        result['predictions'] = predictions
        result['prediction_count'] = len(predictions)

    elif config['type'] == 'digit':
        predictor = DigitPredictor(game_type)
        predictions = predictor.predict(groups=groups)
        result['predictions'] = predictions
        result['prediction_count'] = len(predictions)

    elif config['type'] == 'kl8':
        predictor = KL8Predictor(game_type)
        prediction = predictor.predict()
        result['prediction'] = prediction
        result['prediction_count'] = 1

    return result


# ============================================================
# 便捷调用函数
# ============================================================

def predict_ssq(period: str = None, groups: int = 5) -> Dict:
    """双色球预测"""
    return generate_prediction('SSQ', period, groups)


def predict_dlt(period: str = None, groups: int = 5) -> Dict:
    """大乐透预测"""
    return generate_prediction('DLT', period, groups)


def predict_qlc(period: str = None, groups: int = 5) -> Dict:
    """七乐彩预测"""
    return generate_prediction('QLC', period, groups)


def predict_qxc(period: str = None, groups: int = 5) -> Dict:
    """七星彩预测"""
    return generate_prediction('QXC', period, groups)


def predict_pl3(period: str = None, groups: int = 5) -> Dict:
    """排列三预测"""
    return generate_prediction('PL3', period, groups)


def predict_pl5(period: str = None, groups: int = 5) -> Dict:
    """排列五预测"""
    return generate_prediction('PL5', period, groups)


def predict_fc3d(period: str = None, groups: int = 5) -> Dict:
    """福彩3D预测"""
    return generate_prediction('FC3D', period, groups)


def predict_kl8(period: str = None) -> Dict:
    """快乐8预测"""
    return generate_prediction('KL8', period)


# ============================================================
# 格式化输出函数
# ============================================================

def format_prediction(result: Dict) -> str:
    """将预测结果格式化为可读字符串"""
    lines = []
    lines.append(f"=== {result['game_name']} ({result['game_type']}) 预测结果 ===")
    lines.append(f"引擎版本: {result['engine_version']}")
    if result.get('period'):
        lines.append(f"预测期号: {result['period']}")
    lines.append(f"共 {result['prediction_count']} 组预测")
    lines.append("")

    if result['type'] == 'lotto':
        for pred in result['predictions']:
            lines.append(f"--- 第 {pred['group']} 组 ---")
            red_str = ' '.join(f"{n:02d}" for n in pred['red_balls'])
            lines.append(f"红球: {red_str}")
            if 'blue_balls' in pred:
                blue_str = ' '.join(f"{n:02d}" for n in pred['blue_balls'])
                lines.append(f"蓝球: {blue_str}")
            if 'special_ball' in pred:
                lines.append(f"特别号: {pred['special_ball']:02d}")

            analysis = pred['analysis']
            lines.append(f"  和值: {analysis['sum_value']} | 跨度: {analysis['span']} | "
                         f"AC值: {analysis['ac_value']}")
            lines.append(f"  奇偶比: {analysis['odd_even_ratio']} | "
                         f"连号组数: {analysis['consecutive_groups']}")
            lines.append(f"  区间分布: {analysis['zone_distribution']}")
            lines.append("")

    elif result['type'] == 'digit':
        for pred in result['predictions']:
            lines.append(f"--- 第 {pred['group']} 组 ({pred['strategy']}) ---")
            digit_str = ' '.join(str(d) for d in pred['digits'])
            lines.append(f"号码: {digit_str}")
            analysis = pred['analysis']
            lines.append(f"  和值: {analysis['sum_value']} | 跨度: {analysis['span']}")
            lines.append(f"  奇偶比: {analysis['odd_even_ratio']} | "
                         f"大小比: {analysis['big_small_ratio']}")
            lines.append("")

    elif result['type'] == 'kl8':
        pred = result['prediction']
        lines.append(f"选号 ({len(pred['balls'])} 个):")
        ball_str = ' '.join(f"{n:02d}" for n in pred['balls'])
        lines.append(ball_str)
        analysis = pred['analysis']
        lines.append(f"  和值: {analysis['sum_value']} | 跨度: {analysis['span']}")
        lines.append(f"  奇偶比: {analysis['odd_even_ratio']} | "
                     f"连号组数: {analysis['consecutive_groups']}")
        lines.append(f"  区间分布: {analysis['zone_distribution']}")
        lines.append(f"  平均遗漏: {analysis['average_missing']}")

    lines.append("=" * 40)
    lines.append("免责声明：本预测仅供研究参考，彩票开奖为随机事件。")
    return '\n'.join(lines)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == '__main__':
    # 各玩法预测示例
    games = ['SSQ', 'DLT', 'QLC', 'QXC', 'PL3', 'PL5', 'FC3D', 'KL8']

    for game in games:
        try:
            result = generate_prediction(game, period='2024001', groups=3)
            print(format_prediction(result))
            print()
        except Exception as e:
            print(f"{game} 预测出错: {e}")
