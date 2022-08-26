# coding: utf-8
# @FileName: 1.py
# @Time: 2022/8/26 19:03
# @Author: QHB

"""
使用模拟退火算法求解一元函数极值问题

example_1:
        f(x) = x^3 - 60 * x^2 - 4 * x + 6
    s.t.
        x ∈ [0, 100]

"""

import numpy as np
import matplotlib.pyplot as plt
import math


# 定义目标函数
def function(x):
    y = x ** 3 - 60 * x ** 2 - 4 * x + 6
    return y


def simulated_annealing(current_temperature, final_temperature, x_lb, x_ub, k, lamda):
    """
    模拟退火算法主函数
    Args:
        current_temperature: 初始温度
        final_temperature: 终止温度
        x_lb: x的下限
        x_ub: x的上限
        k: 每个固定温度下的循环次数
        lamda: 温度衰减系数

    Returns:

    """
    # 初始化 x
    x = np.random.uniform(low=x_lb, high=x_ub)
    # 初始解
    y = 0
    while current_temperature >= final_temperature:
        for i in range(k):
            # 计算当前评价值
            y = function(x)
            x_new = -1
            # 保证新生成的解是可行解
            while x_new < 0 or x_new > 100:
                # 生成新的候选解
                x_new = x + np.random.uniform(low=-0.055, high=0.055) * current_temperature
            y_new = function(x_new)
            # 如果产生的新解更优秀, 则选择新解
            if y_new - y < 0:
                x = x_new
            else:
                # 否则根据 metropolis准则 按照一定概率接受坏解
                p = math.exp(-(y_new - y) / current_temperature)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    x = x_new
        # 降温函数
        current_temperature *= lamda
        print("current temperature is: ", str(current_temperature), ", current x value is: ", str(x),
              ", current y value is: ", str(y))
    return x, function(x)


if __name__ == '__main__':
    # 绘图
    x = [i / 10 for i in range(1000)]
    y = [0 for i in range(1000)]
    for i in range(1000):
        y[i] = function(x[i])
    plt.plot(x, y)
    x_res, y_res = simulated_annealing(current_temperature=1000, final_temperature=10, x_lb=0, x_ub=100, k=50, lamda=0.95)
    plt.scatter(x_res, y_res, marker='+', color='coral')
    plt.show()
    print("final x value is: ", str(x_res), ", final y value is: ", str(y_res))
