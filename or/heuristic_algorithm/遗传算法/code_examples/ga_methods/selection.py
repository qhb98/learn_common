from random import random

import numpy as np

"""

遗传算法中常见的选择算子

"""


def selection_roulette(population, population_fitness):
    """
    轮盘赌选择算子
    Args:
        population: 种群
        population_fitness: 种群中每个个体对应的适应度值组成的集合

    Returns: 返回经过轮盘赌选择后的新种群

    """
    # 种群适应度之和
    sum_fitness = sum(population_fitness)
    prob = [population_fitness[i] / sum_fitness for i in range(len(population))]
    chosen = []
    # 轮盘赌的选择次数应当与种群规模一致, 第一次遍历是为了保证选择前后的父代和子代之间的种群规模是完全一致的
    for i in range(len(population)):
        cum = 0
        # 随机概率
        m = random()
        # 轮盘赌选择
        for j in range(len(population)):
            # 概率累加
            cum += prob[j]
            if cum >= m:
                chosen.append(population[j])
                break
    return chosen


def selection_tournament(tour_size=3, population, population_fitness):
    """
    锦标赛选择算子
    在锦标赛选择方法的每一轮中，从总体中随机选择两个或多个个体，其中适应度得分最高的获胜并被选中

    Args:
        tour_size: 每次选择的个体数量
        population: 种群
        population_fitness: 种群中每个个体的适应度值组成的集合

    Returns:

    """
    # 保存选择的个体在原种群中的索引
    select_idx = []
    for i in range(len(population)):
        # aspirants_index = np.random.choice(range(len(population)), size=tour_size)
        # 从种群中随机获取tour_size个个体, randint适用于实数编码, random.choice适用于TSP问题
        aspirants_index = np.random.randint(range(len(population)), size=tour_size)
        select_idx.append(max(aspirants_index, key=lambda i: population_fitness[i]))
    # 根据索引更新种群, 得到下一代种群
    population = population[select_idx, :]
    return population


def selection_tournament_faster(tour_size=3, population, population_fitness):
    """
    和selection_tournament一样, 只是运算性能上加速了 numpy的作用
    Args:
        tour_size:
        population:
        population_fitness:

    Returns:

    """
    aspirants_idx = np.random.randint(len(population), size=(len(population), tour_size))
    aspirants_values = population_fitness[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
    population = population[sel_index, :]
    return population


def selection_stochastic_universal_sampling(population, population_fitness):
    """
    随机遍历抽样 SUS
    随机遍历抽样是先前描述的轮盘选择的修改版本
    使用相同的轮盘，比例相同，但使用多个选择点，只旋转一次转盘就可以同时选择所有个体

    step1 计算指针的间距 p = sum(fitness) / len(population)
    step2 随机生成起点指针位置 start = np.random.randint(p)
    step3 计算各指针的位置 pointers = [start + i * p], i=0,1,...,len(population) - 1
    step4 根据各个指针的位置选择出N个个体

    Args:
        population: 种群
        population_fitness: 种群中每个个体对应的适应度值组成的集合

    Returns:

    """
    # 种群适应度之和
    sum_fitness = sum(population_fitness)
    # prob = [population_fitness[i] / sum_fitness for i in range(len(population))]
    # prob_cum = [sum(prob[:i+1]) for i in range(len(prob))]
    chosen = []
    # 计算指针的间距
    p = sum_fitness / len(population)
    for i in range(len(population)):
        pos_cur = np.random.randint(p, size=1) + i * p
        chosen.append(population[pos_cur, :])
    return chosen
