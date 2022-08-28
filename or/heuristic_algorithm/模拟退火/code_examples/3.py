# coding: utf-8
# @FileName: 3.py
# @Time: 2022/8/26 22:12
# @Author: QHB
# coding:gbk

"""

example_3:
    模拟退火算法求解背包问题

"""

import random
import math


def calc(things_weights, things_values, cur_res):
    """
    计算当前的物品占据的背包的重量和物品的价值
    Args:
        things_weights: 各物品的重量列表
        things_values: 各物品的价值列表
        cur_res: 当前解[0 1 0...]

    Returns:

    """
    bag_values = 0
    bag_weights = 0
    for i in range(len(things_weights)):
        bag_values += things_values[i] * cur_res[i]
        bag_weights += cur_res[i] * things_weights[i]
    return bag_weights, bag_values


def init_res_produce(things_weights, things_values, cur_res, bag_capacity):
    """
    初始化函数, 用于产生随机的初始解
    Args:
        things_weights: 物品的重量列表
        things_values: 物品的价值列表
        cur_res: 当前的解
        bag_capacity: 背包的最大重量

    Returns:

    """
    # 初始产生随机解
    while True:
        for i in range(len(things_weights)):
            if random.random() < 0.5:
                cur_res[i] = 1
            else:
                cur_res[i] = 0
        # 计算初始解的占据的重量和价值
        cur_bag_weight, cur_bag_value = calc(things_weights, things_values, cur_res)
        # 保证产生的是可行解
        if cur_bag_weight < bag_capacity:
            break
    return cur_res[:]


def simulated_annealing(weights, values, cur_tem, min_tem, lamda_value, l_max, cur_res, bag_capacity, best_value):
    """
    模拟退火求解背包问题主函数
    Args:
        weights: 各物品的重量列表
        values: 各物品的价值列表
        cur_tem: 当前温度
        min_tem: 最低温度
        lamda_value: 温度衰减系数
        l_max: 固定温度下循环迭代的最大次数
        cur_res: 当前的解
        bag_capacity: 背包的最大重量
        best_value: 最优解对应的价值

    Returns:

    """
    # 初始化解
    init_res = init_res_produce(weights, values, cur_res, bag_capacity)
    cur_res = init_res
    # 记录历史解
    trend = []
    # 记录最优解
    best_res = None
    while cur_tem >= min_tem:
        for i in range(l_max):
            cur_bag_weight, cur_bag_value = calc(weights, values, cur_res)
            # 随机产生新解 randint(0, 1)生成的是[0, 1]范围内的整数, 注意是双闭区间
            test_res = [random.randint(0, 1) for i in range(len(weights))]
            choose = random.randint(0, len(weights) - 1)
            # 在背包中则将其拿出，并加入其它物品
            if test_res[choose] == 1:
                test_res[choose] = 0
            else:
                # 不在背包中则按照概率直接加入
                if random.random() < 0.5:
                    test_res[choose] = 1
            test_bag_weight, test_bag_value = calc(weights, values, test_res)
            # 非法解则跳过
            if test_bag_weight > bag_capacity:
                continue
            # 如果新解更优, 则更新解, 否则按metropolis准则更新解
            if test_bag_value > cur_bag_value:
                cur_res = test_res[:]
            else:
                # 按概率接受劣解
                if random.random() < math.exp(- (test_bag_value - cur_bag_value) / cur_tem):
                    cur_res = test_res[:]
        # 温度下降
        cur_tem *= lamda_value
        cur_bag_weight, cur_bag_value = calc(weights, values, cur_res)
        # 将当前的局部最优解和已知的全局最优解比较, 若更佳, 则更新全局最优
        if cur_bag_value >= best_value:
            best_value = cur_bag_value
            best_res = cur_res
        trend.append((cur_res, cur_bag_weight, cur_bag_value))
        print(cur_tem, cur_res)
    return trend, best_value, best_res


if __name__ == '__main__':
    # 背包的容量
    C = 8
    # 物品占据的空间
    weight = [2, 3, 5, 1, 4]
    # 物品的价值
    value = [2, 5, 8, 3, 6]
    # 当前温度
    current_tem = 2000
    # 最终的温度
    final_tem = 10
    # 温度衰减系数
    lamda = 0.95
    # 迭代次数
    k = 100
    # 物品数量
    things_number = len(weight)
    # cur_way 记录当前解
    cur_way = [0] * things_number
    # 记录最优解对应的价值
    best_res_value = 0
    res_trend, best_result_value, best_result = simulated_annealing(weight, value, current_tem, final_tem, lamda, k,
                                                                    cur_way, C,
                                                                    best_res_value)
    print("best res is: ", str(best_result))
