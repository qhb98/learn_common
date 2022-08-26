# coding: utf-8
# @FileName: 2.py
# @Time: 2022/8/26 21:17
# @Author: QHB

"""
example_2:
    模拟退火算法求解TSP问题

"""

import random
from math import exp
import matplotlib.pyplot as plt


def distance_calculate(a, b):
    """
    计算任意两个城市之间的距离
    Args:
        a: 城市a
        b: 城市b

    Returns:

    """
    x1 = city_loc[a][0]
    x2 = city_loc[b][0]
    y1 = city_loc[a][1]
    y2 = city_loc[b][1]
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


def total_distance_calculate(cities):
    """
    计算 路程总长
    Args:
        cities: 城市坐标数据

    Returns:

    """
    total_dis = 0
    for j in range(len(cities) - 1):
        total_dis += distance_calculate(cities[j], cities[j + 1])
    # 加上到最初的那个城市之间的距离
    total_dis += distance_calculate(cities[-1], cities[0])
    return total_dis


def init_ans(cites):
    """
    初始化一个解
    Args:
        cites:

    Returns:

    """
    res = []
    for i in range(len(cites)):
        res.append(i)
    return res


def creat_new(ans_before):
    """
    交换操作生成新的候选解
    这儿可以有很多操作生成新的邻域解, 比如逆序 插入等
    Args:
        ans_before: 原解

    Returns:

    """
    cut_a = 0
    cut_b = 0
    ans_after = ans_before[:]
    # 交换操作
    while cut_a == cut_b:
        cut_a = random.randint(0, 30)
        cut_b = random.randint(0, 30)
    ans_after[cut_a], ans_after[cut_b] = ans_before[cut_b], ans_before[cut_a]
    return ans_after


def simulated_annealing(current_tem, final_tem, lamda, l_max, init_res):
    """
    模拟退火核心代码
    Args:
        current_tem: 当前温度
        final_tem: 最终温度
        lamda: 温度衰减系数
        l_max: 固定温度下的最大迭代次数
        init_res: 初始解

    Returns: 返回每个温度下的结果

    """
    trend = []
    cur_res = init_res
    while current_tem > final_tem:
        for i in range(l_max):
            new_ans = creat_new(cur_res)
            old_dist = total_distance_calculate(cur_res)
            new_dist = total_distance_calculate(new_ans)
            if new_dist > old_dist:
                if exp((old_dist - new_dist) / current_tem) > random.uniform(0, 1):
                    cur_res = new_ans
            else:
                cur_res = new_ans
        trend.append((cur_res, total_distance_calculate(cur_res)))
        current_tem *= lamda
    return trend


if __name__ == '__main__':
    # 31个城市的坐标数据
    city_loc = [
        (1304, 2312), (3639, 1315), (4177, 2244), (3712, 1399), (3488, 1535),
        (3326, 1556), (3238, 1229), (4196, 1004), (4312, 790), (4380, 570),
        (3007, 1970), (2562, 1756), (2788, 1491), (2381, 1676), (1332, 695),
        (3715, 1678), (3918, 2179), (4061, 2370), (3780, 2212), (3676, 2578),
        (4029, 2838), (4263, 2931), (3429, 1908), (3507, 2367), (3394, 2643),
        (3439, 3201), (2935, 3240), (3140, 3550), (2545, 2357), (2778, 2826),
        (2370, 2975)
    ]
    # 使用模拟退火计算最佳路径
    results = simulated_annealing(current_tem=50000, final_tem=15, lamda=0.95, l_max=100, init_res=init_ans(city_loc))
    plt.plot([city_loc[i][0] for i in min(results[0], key=lambda x: results[1])], [city_loc[i][1] for i in min(results[0], key=lambda x: results[1])])
    plt.show()
