# coding: utf-8
# @FileName: 2.py
# @Time: 2022/8/28 18:26
# @Author: QHB

"""

example_2:
    使用蚁群算法求解TSP问题

"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ACO(object):
    def __init__(self, num_city, data):
        # 蚂蚁数量
        self.m = 50
        # 信息素重要程度因子
        self.alpha = 0.5
        # 启发函数重要因子
        self.beta = 5
        # 信息素挥发因子
        self.rho = 0.1
        # 常量系数
        self.Q = 1
        # 城市规模
        self.num_city = num_city
        # 城市坐标
        self.location = data
        # 信息素矩阵
        self.Tau = np.zeros([num_city, num_city])
        # 生成的蚁群
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]
        self.iter = 1
        self.iter_max = 500
        # 计算城市之间的距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, self.location)
        # 启发式函数
        self.Eta = 10. / self.dis_mat
        # 蚁群中每个个体的长度
        self.paths = None
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = []
        self.iter_y = []
        self.greedy_init(self.dis_mat, 100, num_city)

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        # argsort()是将X中的元素从小到大排序后, 提取对应的索引index,然后输出到y
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        result = result[index]
        for i in range(len(result) - 1):
            s = result[i]
            s2 = result[i + 1]
            self.Tau[s][s2] = 1
        self.Tau[result[-1]][result[0]] = 1
        # for i in range(num_city):
        #     for j in range(num_city):
        # return result

    # 轮盘赌选择
    def rand_choose(self, p):
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    # 生成蚁群
    def get_ants(self, num_city):
        for i in range(self.m):
            start = np.random.randint(num_city - 1)
            self.Table[i][0] = start
            unvisit = list([x for x in range(num_city) if x != start])
            current = start
            j = 1
            while len(unvisit) != 0:
                P = []
                # 通过信息素计算城市之间的转移概率
                for v in unvisit:
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                # 轮盘赌选择一个一个城市
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算一条路径的长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        paths = self.compute_paths(self.Table)
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        best_lenth = math.inf  # math.inf返回浮点正无穷大
        best_path = None
        for cnt in range(self.iter_max):
            # 生成新的蚁群
            self.get_ants(self.num_city)  # out>>self.Table
            self.paths = self.compute_paths(self.Table)
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_lenth)]
            # 可视化初始的路径
            if cnt == 0:
                init_show = self.location[tmp_path]
                init_show = np.vstack([init_show, init_show[0]])
            # 更新最优解
            if tmp_lenth < best_lenth:
                best_lenth = tmp_lenth
                best_path = tmp_path
            # 更新信息素
            self.update_Tau()

            # 保存结果
            self.iter_x.append(cnt)
            self.iter_y.append(best_lenth)
            print(cnt, best_lenth)
        return best_lenth, best_path

    def run(self):
        best_length, best_path = self.aco()
        return self.location[best_path], best_length


# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    source_data = []
    for line in [line.strip("\n").split(" ") for line in lines]:
        source_data.append([int(line[0]), int(line[1]), int(line[2])])
    return np.array(source_data)


if __name__ == '__main__':
    data = read_tsp(r'../datasets/tsp70.txt')
    data = data[:, 1:]
    # 加上一行因为会回到起点
    show_data = np.vstack([data, data[0]])

    aco = ACO(num_city=data.shape[0], data=show_data.copy())
    Best_path, Best = aco.run()
    print(Best)
    Best_path = np.vstack([Best_path, Best_path[0]])
    plt.plot(Best_path[:, 0], Best_path[:, 1])
    plt.title('aco-tsp results of 70 cities')
    plt.show()
