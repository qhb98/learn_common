# coding: utf-8
# @FileName: 1.py
# @Time: 2022/8/25 17:52
# @Author: QHB

"""

使用随机爬山算法求解 函数function = sin(x^2) + 2*cos(2*x) 在区间[5,8]的最大值

"""

import numpy as np
import matplotlib.pyplot as plt
import math


def function(x):
    return math.sin(x * x) + 2.0 * math.cos(2.0 * x)


def hill_climbing(x, BOUND, DELTA):
    """
    爬山搜索
    Args:
        x:

    Returns:

    """
    while function(x + DELTA) > function(x) and BOUND[1] >= x + DELTA >= BOUND[0]:
        x += DELTA
    while function(x - DELTA) > function(x) and BOUND[1] >= x - DELTA >= BOUND[0]:
        x -= DELTA
    return x, function(x)


def find_max(BOUND, DELTA, GENERATION):
    # 用于记录搜索到的最优解的 x和对应的function(x)
    highest = [0, -1000]
    # 循环迭代generation次, 不断寻找最优解
    for i in range(GENERATION):
        x = np.random.rand() * (BOUND[1] - BOUND[0]) + BOUND[0]
        current_value = hill_climbing(x, BOUND, DELTA)
        print('current value is :', current_value)

        if current_value[1] > highest[1]:
            highest[:] = current_value
    return highest


# 搜索步长
DELTA = 0.01
# 定义域x从5到8闭区间
BOUND = [5, 8]
# 随机取乱数100次
GENERATION = 100
[x, y] = find_max(BOUND, DELTA, GENERATION)

print('highest point is x :{},y:{}'.format(x, y))
