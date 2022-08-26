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


def cop(a, b, le):
    # 复制函数 把b数组的值赋值a数组
    for i in range(le):
        a[i] = b[i]






def get(x):
    # 随机将背包中已经存在的物品取出
    while 1 > 0:
        ob = random.randint(0, m - 1)
        if x[ob] == 1:
            x[ob] = 0
        break


def put(x):  # 随机放入背包中不存在的物品
    while 1 > 0:
        ob = random.randint(0, m - 1)
        if x[ob] == 0:
            x[ob] = 1
        break


def slove():  # 迭代函数
    global best, T, balance
    test = [0] * m
    now = 0  # 当前背包价值
    for i in range(balance):
        now = calc(now_way)
        cop(test, now_way, m)
        ob = random.randint(0, m - 1)  # 随机选取某个物品
        if (test[ob] == 1):
            put(test)
            test[ob] = 0  # 在背包中则将其拿出，并加入其它物品
        else:  # 不在背包中则直接加入或替换掉已在背包中的物品
            if (random.random() < 0.5):
                test[ob] = 1
            else:
                get(test)
                test[ob] = 1
        temp = calc(test)
        if (wsum > C): continue  # 非法解则跳过
        if temp > best: best = temp
        cop(best_way, test, m)  # 更新全局最优

        if (temp > now):
            cop(now_way, test, m)  # 直接接受新解 
        else:
            g = 1.0 * (temp - now) / T
            if (random.random() < math.exp(g)):  # 概率接受劣解
                cop(now_way, test, m)


def calc(things_weights, things_values, cur_res):
    # 计算背包当前的重量和价值
    bag_values = 0
    bag_weights = 0
    for i in range(len(things_weights)):
        bag_values += things_values[i] * cur_res[i]
        bag_weights += cur_res[i] * things_weights[i]
    return bag_weights, bag_values


def init_res_produce(things_weights, things_values, cur_res, best_res):
    # 初始产生随机解
    while 1 > 0:
        for i in range(len(things_weights)):
            if random.random() < 0.5:
                cur_res[i] = 1
            else:
                cur_res[k] = 0
        # 计算初始解的价值
        cur_bag_weight, cur_bag_value = calc(things_weights, things_values, cur_res)
        if wsum < C:
            break
    best = calc(cur_res)
    cop(best_way, cur_res, len(things_weights))


def simulated_annealing(weights, values, cur_tem, lamda_value, l_max, balance_number, best_res, cur_res):
    # 初始化
    init_res_produce(weights, values, cur_res, best_res)
    isGood = 0
    for i in range(k_max):
        slove()
        T = T * af  # 温度下降
        if best == 295:
            print('找到最优解:295,迭代次数', i + 1)
            isGood = 1
            break  # 达到最优解提前退出

    if isGood == 0:
        print('只找到次优解:', best, '迭代次数', k_max)
    print('方案为：', best_way)  # 打印方案


if __name__ == '__main__':
    # 物品占据的空间
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    # 物品的价值
    value = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    # 当前温度
    current_tem = 200.0
    # 温度衰减系数
    lamda = 0.95
    # 迭代次数
    k = 10
    # 温度平衡的最大次数
    balance_num = 100
    # 物品数量
    things_number = len(weight)
    # best_way 记录全局最优解
    best_way = [0] * things_number
    # cur_way 记录当前解
    cur_way = [0] * things_number
    simulated_annealing(weight, value, current_tem, lamda, k, balance_num, best_way, cur_way)
