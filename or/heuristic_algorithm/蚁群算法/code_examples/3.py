# coding: utf-8
# @FileName: 3.py
# @Time: 2022/8/28 18:26
# @Author: QHB

"""

example_3:
    使用蚁群算法优化函数, 求解最大值
            y = 4 * x1 ** 2 + 2 * x2 + x3 ** 3
               s.t.
                 x1, x2, x3 = [1, 30]

"""

import numpy as np
import matplotlib.pyplot as plt


class ACO:
    def __init__(self, parameters):
        # 初始化
        self.NGEN = parameters[0]
        # 种群的规模
        self.pop_size = parameters[1]
        # 变量的个数
        self.var_num = len(parameters[2])
        # 变量的约束范围
        self.bound = []
        # 下限和上限
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        # 所有蚂蚁的位置
        self.pop_x = np.zeros((self.pop_size, self.var_num))
        # 全局蚂蚁最优的位置
        self.g_best = np.zeros((1, self.var_num))
        # 初始化解的方法
        self.init_ants()

    def init_ants(self):
        """
        生成初始解
        Returns:

        """
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            # 计算第i只蚂蚁的适应度值
            fit = ACO.fitness_cal(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit

    @staticmethod
    def fitness_cal(ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]

        y = 4 * x1 ** 2 + 2 * x2 + x3 ** 3
        return y

    def update_operator(self, gen, t, t_max):
        """
        更新算子：根据概率更新下一时刻的位置

        Args:
            gen: 当前迭代到的代数
            t: 当前种群内的所有蚂蚁的适应度值
            t_max: 最大的适应度值

        Returns:

        """
        # 信息素挥发系数
        rou = 0.8
        # 信息释放总量
        Q = 1
        #
        lamda = 1 / gen
        # 记录信息素
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                # 计算信息素转移概率
                pi[i] = (t_max - t[i]) / t_max
                # 根据信息素转移概率计算下一时刻的位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] += np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] += np.random.uniform(-1, 1) * (self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护, 考虑变量的约束限制
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 信息素更新
            t[i] = (1 - rou) * t[i] + Q * ACO.fitness_cal(self.pop_x[i])
            # 更新全局最优值
            if ACO.fitness_cal(self.pop_x[i]) > ACO.fitness_cal(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def run(self):
        pop_obj = []
        best = np.zeros((1, self.var_num))[0]
        # 迭代的区间范围在[1, NGEN+1]是考虑到 lambda = 1 / gen
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                fit_value_max, fit_values = self.update_operator(gen, np.array(list(map(ACO.fitness_cal, self.pop_x))),
                                                np.max(np.array(list(map(ACO.fitness_cal, self.pop_x)))))
            else:
                fit_value_max, fit_values = self.update_operator(gen, fit_values, fit_value_max)

            print('current generation is {}'.format(str(gen)))
            print('current best variables are {}'.format(str(self.g_best)))
            print('current best fitness value is {}'.format(str(ACO.fitness_cal(self.g_best))))
            if ACO.fitness_cal(self.g_best) > ACO.fitness_cal(best):
                best = self.g_best.copy()
            pop_obj.append(ACO.fitness_cal(best))
            print('best variables values {}.'.format(best))
            print('max function values {}.'.format(ACO.fitness_cal(best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        plt.plot([t for t in range(1, self.NGEN + 1)], pop_obj, color='b', linewidth=2)
        plt.show()


if __name__ == '__main__':
    # 最大迭代次数
    NGEN = 100
    # 蚂蚁种群的规模
    pop_size = 50
    # 下限
    low = [1, 1, 1]
    # 上限
    up = [30, 30, 30]
    # 输入参数
    parameters = [NGEN, pop_size, low, up]
    # 实例化对象
    aco = ACO(parameters)
    # 运行方法
    aco.run()
