# coding: utf-8
# @FileName: :tabu.py
# @Time: 2022/8/6 14:44
# @Author: QHB

from itertools import combinations
import os, sys, copy
import numpy as np
import time
import matplotlib.pyplot as plt
from data_reader import Data


class Tabu:
    """
    源自清华运小筹的整理笔记, 但个人认为代码略有问题, 所有认为可能有问题的地方均打上了 todo
    标记, 留待后续处理

    另, 原版的代码对于变量命名、代码规范等还有待商榷, 此.py文件已做了简单的优化处理
    """
    def __int__(self, dis_matrix, max_iters=50, max_tabu_size=10):
        """
        定义需要计算的参数
        """
        self.dis_matrix = dis_matrix
        self.max_iters = max_iters
        self.maxTabuSize = max_tabu_size
        self.tabu_list = []

    def get_route_distance(self, route):
        """
        计算route的全部距离长度的评价函数

        """
        routes = [0] + route + [0]
        total_distance = 0
        for i, n in enumerate(routes):
            if i != 0:
                # todo
                total_distance = total_distance + self.dis_matrix[last_pos][n]
            last_pos = n
        return total_distance

    @staticmethod
    def exchange(s1, s2, arr):
        """
        将arr中的两个elements交换位置
        Args:
            s1: target 1
            s2: target 2
            arr: target array

        Returns: new target array

        """
        current_list = copy.deepcopy(arr)
        index1, index2 = current_list.index(s1), current_list.index(s2)
        current_list[index1], current_list[index2] = arr[index2], arr[index1]
        return current_list

    def generate_initial_solution(self, num=10, mode="greedy"):
        """
        获得初始解的方法, 有两种不同的方式去生成route_init
        Args:
            num: 点位的数量
            mode: 模式, 规则选择

        Returns: 经过初始化后得到的route_init

        """
        route_init = None
        if mode == "greedy":
            route_init = [0]
            for i in range(num):
                best_distance = 10000000
                best_candidate = None
                for j in range(num + 1):
                    if self.dis_matrix[i][j] < best_distance and j not in route_init:
                        best_distance = self.dis_matrix[i][j]
                        best_candidate = j
                route_init.append(best_candidate)
            route_init.remove(0)

        elif mode == "random":
            # 初始化
            route_init = np.arange(1, num + 1)
            # 随机打乱当前的list
            np.random.shuffle(route_init)

        return list(route_init)

    def tabu_search(self, s_init):
        s_best = s_init
        best_candidate = copy.deepcopy(s_best)
        # 初始化
        routes, temp_tabu = [], []
        routes.append(s_best)
        while self.max_iters:
            # 迭代的次数
            self.max_iters -= 1
            neighbors = copy.deepcopy(s_best)
            for s in combinations(neighbors, 2):
                # 交换数值生成candidates
                s_candidate = Tabu.exchange(s[0], s[1], neighbors)
                if s not in self.tabu_list and self.get_route_distance(s_candidate) < self.get_route_distance(
                        best_candidate):
                    best_candidate = s_candidate
                    temp_tabu = s
            if self.get_route_distance(best_candidate) < self.get_route_distance(s_best):
                s_best = best_candidate
            elif temp_tabu not in self.tabu_list:
                self.tabu_list.append(temp_tabu)
            elif len(self.tabu_list) > self.maxTabuSize:
                self.tabu_list.pop(0)
            routes.append(best_candidate)
        return s_best, routes


if __name__ == '__main__':
    data = Data()
    tsp = Tabu()

    data.read_data(path="R101.txt", customer_num=100, depot_num=1)
    s_init = tsp.generate_initial_solution(num=10, mode="greedy")
    print("init route: ", s_init)
    print("init distance:", tsp.get_route_distance(s_init))

    start_time = time.time()
    best_route, routes = tsp.tabu_search(s_init)
    end_time = time.time()
    # 绘图
    results = []
    for i in routes:
        results.append(tsp.get_route_distance(i))
    plt.plot(np.arange(len(results), results))
    plt.show()
    data.plt_route(best_route)
