# coding: utf-8
# @FileName: 1.py
# @Time: 2022/8/29 19:34
# @Author: QHB

"""
example_1:
    使用差分进化算法求解函数优化问题
        f(x) = x1 ** 2 + x2 ** 2
        s.t.
            x1 ∈ (-1, 1)
            x2 ∈ (-1, 1)

"""

import numpy as np
import random


class DifferentialEvolution:
    def __init__(self, min_range, max_range, dim, factor, rounds, size, object_func, CR=0.75):
        # 变量的下限
        self.min_range = min_range
        # 变量的上限
        self.max_range = max_range
        # 变量个数
        self.dimension = dim
        #
        self.factor = factor
        #
        self.rounds = rounds
        #
        self.size = size
        #
        self.cur_round = 1
        # 变异率
        self.CR = CR
        # 输入待优化的函数
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range)
                                        for s in range(self.dimension)]) for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)

    def crossover_and_select(self):
        for i in range(self.size):
            j_rand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > self.CR and j != j_rand:
                    self.mutant[i][j] = self.individuality[i][j]
                tmp = self.get_object_function_value(self.mutant[i])
                if tmp < self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tmp

    def print_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(self.individuality[i]))
        print("目标函数值：" + str(m))

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            self.print_best()
            self.cur_round = self.cur_round + 1


def f(v):
    return -(v[1] + 47) * np.sin(np.sqrt(np.abs(v[1] + (v[0] / 2) + 47))) - v[0] * np.sin(
        np.sqrt(np.abs(v[0] - v[1] - 47)))


# 测试部分
if __name__ == "__main__":
    p = DifferentialEvolution(min_range=-513, max_range=513, dim=2, factor=0.8, rounds=100, size=100, object_func=f)
    p.evolution()
