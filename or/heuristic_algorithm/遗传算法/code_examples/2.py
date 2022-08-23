# coding: utf-8
# @FileName: 2.py
# @Time: 2022/8/23 15:07
# @Author: QHB
"""
example_2:
非线性函数最优化问题求解_二元 ———— 实数编码方式

"""

import math
from random import random, randint, uniform
from matplotlib import pyplot as plt


# ============================== 定义非线性函数 =================================== #
def func(x, y):
    numerator = 6.452 * (x + 0.125 * y) * (math.cos(x) - math.cos(2 * y)) ** 2
    denominator = math.sqrt(0.8 + (x - 4.2) ** 2 + 2 * (y - 7) ** 2)
    return numerator / denominator + 3.226 * y


# ============================== GA类 ============================================ #
class GeneticAlgorithm:
    def __init__(self, function, population, gen, pc=0.85, pm=0.05):
        """
        :param function: 适应度函数/目标函数
        :param population: 种群
        :param gen: 总迭代次数
        :param pc: 交叉概率
        :param pm: 变异概率
        """
        # 定义函数
        self.func = function
        # 交叉概率
        self.pc = pc
        # 变异概率
        self.pm = pm
        # 种群规模
        self.population = population
        # 最大迭代次数
        self.gen = gen
        # 浮点数的精度 小数点后3位数
        self.dec_num = 3
        # 存储 决策变量X Y
        self.X = []
        self.Y = []
        # 存储整个种群的染色体
        self.chr = []
        # 适应度值
        self.f = []
        # 适应度值排序后的结果
        self.rank = []
        # 存储每代的历史数据
        self.history = {
            'f': [],
            'x': [],
            'y': []
        }

    def num2str(self, num):
        """
        将浮点数变成string
        Args:
            num: 将数字转换为实数

        Returns:

        """
        # return str(int(num)) + str(int(num - int(num)) * 1e3)
        s = str(num).replace('.', '')
        s += '0' * abs(int(num) // 10 + 1 + self.dec_num - len(s))
        return s

    def encoder(self, x, y):
        """
        实数编码, 将x y两个决策变量的值转换为字符串后拼接在一起组成染色体, 一条染色体即为一个个体
        Args:
            x: 决策变量x的值
            y: 决策变量y的值

        Returns:

        """
        chr_list = []
        for i in range(len(x)):
            chr = self.num2str(x[i]) + self.num2str(y[i])
            chr_list.append(chr)
        return chr_list

    def str2num(self, s):
        """
        解码
        Args:
            s: 字符串转换为数值

        Returns:

        """
        num = int(s[:-self.dec_num]) + float(s[-self.dec_num:]) / 10 ** self.dec_num
        return round(num, self.dec_num)

    def decoder(self, chr):
        """
        解码
        Args:
            chr:

        Returns:

        """
        cut = int(len(chr[0]) / 2)
        x = [self.str2num(chr[i][:cut]) for i in range(len(chr))]
        y = [self.str2num(chr[i][cut:]) for i in range(len(chr))]
        return x, y

    def choose(self):
        # 计算
        s = sum(self.f)
        p = [self.f[i] / s for i in range(self.population)]
        chosen = []
        # 轮盘赌的选择次数应当与种群规模一致, 保证选择前后的父代和子代之间的种群规模是完全一致的
        for i in range(self.population):
            cum = 0
            m = random()
            # 轮盘赌选择
            for j in range(self.population):
                cum += p[j]
                if cum >= m:
                    chosen.append(self.chr[j])
                    break
        return chosen

    def crossover(self, chr):
        """
        单点交叉算子
        Args:
            chr:

        Returns:

        """
        crossed = []
        # 如果种群规模是奇数, 就将最后一个pop出来直接给下一代
        if len(chr) % 2:
            crossed.append(chr.pop())
        # 步长为2的原因是: 对当代的种群做两两交叉, 需要当代种群提供 父本和母本
        for i in range(0, len(chr), 2):
            a = chr[i]
            b = chr[i + 1]
            # 交叉的概率为0.85
            if random() < self.pc:
                # 采用的是单点交叉的方法
                loc = randint(1, len(chr[i]) - 1)
                temp = a[loc:]
                a = a[:loc] + b[loc:]
                b = b[:loc] + temp
            # 添加到交叉后的子代种群中
            crossed.append(a)
            crossed.append(b)
        return crossed

    def mutation(self, chr):
        """
        基本位变异算子
        Args:
            chr: 种群

        Returns:

        """
        res = []
        for i in chr:
            l = list(i)
            # 选择了1个或多个基因进行变异
            for j in range(len(l)):
                # 变异概率为0.05
                if random() < self.pm:
                    while True:
                        r = str(randint(0, 9))
                        if r != l[j]:
                            l[j] = r
                            break
            res.append(''.join(l))
        return res

    def run(self):
        """
        运行接口
        Returns:

        """
        # 初始化
        x = []
        y = []
        for i in range(self.population):
            x.append(round(uniform(0, 10), self.dec_num))
            y.append(round(uniform(0, 10), self.dec_num))
        self.X = x
        self.Y = y
        self.chr = self.encoder(x, y)
        # 循环迭代
        for iter in range(self.gen):
            # 计算适应度值
            self.f = [func(self.X[i], self.Y[i]) for i in range(self.population)]
            fitness_sort = sorted(enumerate(self.f), key=lambda x: x[1], reverse=True)
            # 排序
            self.rank = [i[0] for i in fitness_sort]
            winner = self.f[self.rank[0]]
            print(f'Iter={iter + 1}, Max-Fitness={winner}')
            # 存储每代最佳的个体
            self.history['f'].append(winner)
            self.history['x'].append(self.X[self.rank[0]])
            self.history['y'].append(self.Y[self.rank[0]])
            # 选择, 交叉, 变异
            chosen = self.choose()
            crossed = self.crossover(chosen)
            self.chr = self.mutation(crossed)
            self.X, self.Y = self.decoder(self.chr)


# ============================== main函数 ============================ #
if __name__ == '__main__':
    # 实例化对象
    ga = GeneticAlgorithm(func, 10, 100)
    # 运行
    ga.run()
    # 绘图
    plt.plot(ga.history['f'])
    plt.title('Fitness value')
    plt.xlabel('Iter')
    plt.show()
