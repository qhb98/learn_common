# coding: utf-8
# @FileName: 2.py
# @Time: 2022/8/28 18:26
# @Author: QHB

"""

example_2:
    使用蚁群算法求解TSP问题

"""

import math
import numpy as np
import matplotlib.pyplot as plt


class ACO(object):
    def __init__(self, num_city, data):
        # 蚂蚁数量
        self.m = 50
        # 信息素重要程度因子: 值越大, 则蚂蚁选择之前走过的路径可能性就越大; 值越小, 则蚁群搜索范围就会减少, 容易陷入局部最优
        self.alpha = 0.5
        # 启发函数重要因子: 值越大, 蚁群越就容易选择局部较短路径, 这时算法收敛速度会加快, 但是随机性不高, 容易得到局部的相对最优
        self.beta = 5
        # 信息素挥发因子
        self.rho = 0.1
        # 常量系数
        self.Q = 1
        # 城市规模
        self.num_city = num_city
        # 城市坐标
        self.location = data
        # 信息素矩阵, 初始化时候都为0, 即均相等
        self.Tau = np.zeros([num_city, num_city])
        # 生成的蚁群
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]
        # 记录当前的迭代次数
        self.iter = 1
        # 记录最大的迭代次数
        self.iter_max = 500
        # 计算城市之间的距离矩阵
        self.dis_mat = self.calculate_dis_mat(num_city, self.location)
        # 启发式函数
        self.Eta = 1. / self.dis_mat
        # 蚁群中每个个体的长度
        self.paths = None
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = []
        self.iter_y = []
        # 生成初始解的方法
        self.greedy_init(self.dis_mat, 100, num_city)

    def greedy_init(self, dis_mat, num_total, num_city):
        """
        生成初始化的解集, 从初始化的解集中筛选出最优的解作为初始解
        生成初始解集的过程基于的是贪婪算法的思想, 贪婪的原则是 距离最近的城市
        这部分事实上可以进一步优化, 使用其他的启发式算法生成初始解, 即 基于元启发式算子的超启发式算法

        Args:
            dis_mat: 城市之间的距离矩阵
            num_total: 随机初始化时生成的初始解的总数, 这个值随意定即可
            num_city: 城市的数量

        Returns:

        """
        # 计数, 0到99
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 允许初始生成的解集中存在重复的解
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 贪婪思想: 直接找距离当前城市最近的城市, 添加到解集中
            result_one = [current]
            # 保证每个解中所有城市都被添加过一遍
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
            # 将当前生成的新解添加到初始解集中
            result.append(result_one)
            start_index += 1
        # 计算路径长度
        path_length = self.compute_paths(result)
        # argsort(x)是将x中的元素从小到大排序后, 提取对应的索引index, 然后输出到y
        sort_index = np.argsort(path_length)
        # 找出贪婪算法求解的初始化解集中的最优解作为真正的初始解
        index = sort_index[0]
        result = result[index]
        # Tau禁忌表记录初始解的情况
        for i in range(len(result) - 1):
            s = result[i]
            s2 = result[i + 1]
            self.Tau[s][s2] = 1
        self.Tau[result[-1]][result[0]] = 1

    def randomly_choose(self, p):
        """
        轮盘赌选择
        Args:
            p:

        Returns:

        """
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    def get_ants(self, num_city):
        """
        生成新的蚁群
        Args:
            num_city: 城市的数量

        Returns:

        """
        for i in range(self.m):
            start = np.random.randint(num_city - 1)
            self.Table[i][0] = start
            unvisit = [x for x in range(num_city) if x != start]  # 保存没有访问过的城市列表
            current = start
            j = 1
            while len(unvisit) != 0:
                P = []
                # 根据信息素计算城市之间的转移概率
                for v in unvisit:
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                # 轮盘赌选择一个一个城市
                index = self.randomly_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1

    def calculate_dis_mat(self, num_city, location):
        """
        计算不同城市之间的距离
        Args:
            num_city: 城市数量
            location: 城市 x y 坐标

        Returns:

        """
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

    def calculate_path_length(self, path, dis_mat):
        """
        计算一条路径的长度
        Args:
            path: 当前的路径
            dis_mat: 距离矩阵

        Returns:

        """
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        """
        计算一个群体内的所有蚂蚁的路径长度
        Args:
            paths:

        Returns:

        """
        result = []
        for path in paths:
            length = self.calculate_path_length(path, self.dis_mat)
            result.append(length)
        return result

    # todo 蚁量模型
    def update_Tau_quantity(self):
        pass

    # todo 蚁密模型
    def update_Tau_density(self):
        pass

    def update_Tau_cycle(self):
        """
        更新信息素
        Returns:

        """
        #
        delta_tau = np.zeros([self.num_city, self.num_city])
        paths = self.compute_paths(self.Table)
        # 遍历每一只蚂蚁, 每只蚂蚁表示一个可行解
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                # 蚁周公式
                delta_tau[a][b] += self.Q / paths[i]
            # TSP问题需要走完最后一个城市后需要回到最初的那个城市
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] += self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        """
        主方法
        Returns:

        """
        # math.inf返回浮点正无穷大
        best_length = math.inf
        best_path = None
        for cnt in range(self.iter_max):
            # 生成新的蚁群
            self.get_ants(self.num_city)
            self.paths = self.compute_paths(self.Table)
            # 取该蚁群的最优解
            tmp_length = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_length)]
            # 可视化初始的路径
            if cnt == 0:
                init_show = self.location[tmp_path]
                init_show = np.vstack([init_show, init_show[0]])
            # 更新最优解
            if tmp_length < best_length:
                best_length = tmp_length
                best_path = tmp_path
            # 更新信息素
            self.update_Tau_cycle()

            # 保存结果
            self.iter_x.append(cnt)
            self.iter_y.append(best_length)
            print(cnt, best_length)
        return best_length, best_path

    def run(self):
        """
        运行
        Returns:

        """
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
