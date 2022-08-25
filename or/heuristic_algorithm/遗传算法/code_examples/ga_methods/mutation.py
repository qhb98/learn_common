# coding: utf-8
# @FileName: mutation.py
# @Time: 2022/8/24 19:40
# @Author: QHB
"""

保存常见的 遗传算法的变异算子 源码, 帮助理解各种变异算子的原理

"""

from random import randint, random

import numpy as np


def mutation_real_number_encoder(population, pm=0.05):
    """
    基本位变异算子
    适用于实数编码
    Args:
        population: 种群

    Returns:

    """
    res = []
    # 遍历种群中的每个个体(该代码只针对单染色体的情况)
    for chrosome in population:
        l = list(chrosome)
        # 选择了1个或多个基因进行变异
        for j in range(len(l)):
            # 变异概率为0.05
            if random() < pm:
                # while true 是为了保证变异前后的基因是不一致的
                while True:
                    r = str(randint(0, 9))
                    if r != l[j]:
                        l[j] = r
                        break
        res.append(''.join(l))
    return res


def mutation_binary_encoder(population, pm=0.05):
    """
    位翻转变异 flip bit mutation
    适用于二进制的编码方式
    ^= 运算的效率非常高, 加速算法的运算性能
    Args:
        population: 种群
        pm: 变异概率

    Returns:

    """
    #
    len_chrom = len(population[0])
    # 生成len(population) * len_chrom 的随机样本, 再与变异概率比较, 生成0 1的二维数组
    mask = (np.random.rand(len(population), len_chrom) < pm)
    # 异或运算
    population ^= mask
    return population


def mutation_tsp_1(population, n_dim=100, prob_mut=0.01):
    """
    基本位变异
    适用于TSP问题
    Args:
        population: 种群
        prob_mut: 变异概率
        n_dim: 城市的数量

    Returns: 返回变异后的种群

    """
    # 种群规模
    size_pop = len(population)
    # 染色体的长度 如果TSP需要回到初始起点, 则n_dim + 1 = len_chrom; 如果TSP不需要回到初始起点, 则n_dim = len_chrom
    len_chrom = len(population[0])
    # 遍历每个个体
    for i in range(size_pop):
        # 变异一个或多个基因位
        for j in range(n_dim):
            if np.random.rand() < prob_mut:
                # 为什么前面是遍历n_dim, 这儿是遍历len_chrom:
                # 因为n_dim <= len_chrom, 避免变异出现index out of range的问题
                n = np.random.randint(0, len_chrom, 1)
                population[i, j], population[i, n] = population[i, n], population[i, j]
    return population


def swap(population):
    """
    交换变异算子
    适用于实数编码
    Args:
        population: 种群

    Returns:

    """
    # 遍历种群中的每个个体
    for individual in population:
        n1, n2 = 0, 0
        # 保证n1 n2的相对顺序正确
        while n1 >= n2:
            # 取交换的变异基因位点
            n1, n2 = np.random.randint(0, len(individual) - 1, 2)
            n1, n2 = n2, n1 + 1
        # 交换基因
        individual[n1], individual[n2] = individual[n2], individual[n1]
    return population


def reverse(population):
    """
    反转变异算子
    适用于实数编码或者TSP VRP问题
    也称为 2-opt算子

    Karan Bhatia, "Genetic Algorithms and the Traveling Salesman Problem", 1994
    https://pdfs.semanticscholar.org/c5dd/3d8e97202f07f2e337a791c3bf81cd0bbb13.pdf

    Args:
        population: 种群

    Returns:

    """
    for individual in population:
        n1, n2 = 0, 0
        # 保证n1 n2的相对顺序正确
        while n1 >= n2:
            n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
            n1, n2 = n2, n1 + 1
        # [::-1] 顺序取反操作
        individual[n1:n2] = individual[n1:n2][::-1]
    return population


def transpose(population):
    """
    适用于实数编码

    Args:
        population: 种群

    Returns:

    """
    new_population = []
    for individual in population:
        # 随机生成三个打断的索引点位
        n1, n2, n3 = sorted(np.random.randint(0, len(individual) - 2, 3))
        n2 += 1
        n3 += 2
        slice1, slice2, slice3, slice4 = individual[0:n1], individual[n1:n2], individual[n2:n3 + 1], individual[n3 + 1:]
        # 重新拼接成新的个体
        new_population.append(np.concatenate([slice1, slice3, slice2, slice4]))
    return new_population
