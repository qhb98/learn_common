# coding: utf-8
# @FileName: encoder.py
# @Time: 2022/8/24 19:40
# @Author: QHB

"""

保存常见的 遗传算法的交叉算子 源码, 帮助理解各种交叉算子的原理

"""
from random import random, randint


def crossover_single_point(population, pc=0.85):
    """
    单点交叉算子
    单点交叉又称为简单交叉, 指在编码个体中只随机设置一个交叉点, 然后在该点相互交换两个配对个体的部分染色体
    单点交叉的特点: 若邻接基因座之间的关系能提供较好的个体性状和较高的个体适应度, 则这种单点交叉操作破坏这种个体性状和降低个体适应度的可能性最小

    Args:
        population: 种群
        pc: 交叉概率

    Returns: 返回经过单点交叉后的新种群

    """
    # 经过单点交叉后的新种群, 即 下一代
    crossed_population = []
    # 如果种群规模是奇数, 就将最后一个pop出来直接给下一代
    if len(population) % 2:
        crossed_population.append(population.pop())
    else:
        # 步长为2的原因: 生成父代和母代个体用于互相交换染色体
        for i in range(0, len(population), 2):
            father = population[i]
            mother = population[i + 1]
            # 若生成的随机数小于交叉概率, 则可以发生交叉, 其实就是模拟父代和母代的交配过程
            if random() < pc:
                # 染色体长度, 这儿只针对单染色体的情形, 双/多染色体会更复杂点, 但其实原理都一样
                len_father = len(father)
                # 随机选择交叉的点位
                loc = randint(1, len_father - 1)
                # 染色体交叉
                temp = father[loc:]
                father = father[:loc] + mother[loc:]
                mother = mother[:loc] + temp
            # 添加到交叉后的子代种群中
            crossed_population.append(father)
            crossed_population.append(mother)
    return crossed_population


def crossover_two_points(population, pc=0.85):
    """
    双点交叉算子
    双点交叉是指在个体编码中随机设置了两个交叉点, 然后再进行部分基因的交换

    一般不太使用多点交叉算子, 其可能破坏良好模式, 影响算法的性能
    Args:
        population: 种群
        pc: 交叉概率

    Returns:

    """
    # 经过双点交叉后的新种群, 即 下一代
    crossed_population = []
    # 如果种群规模是奇数, 就将最后一个pop出来直接给下一代
    if len(population) % 2:
        crossed_population.append(population.pop())
    else:
        # 步长为2的原因: 生成父代和母代个体用于互相交换染色体
        for i in range(0, len(population), 2):
            father = population[i]
            mother = population[i + 1]
            # 若生成的随机数小于交叉概率, 则可以发生交叉, 其实就是模拟父代和母代的交配过程
            if random() < pc:
                # 染色体长度, 这儿只针对单染色体的情形, 双/多染色体会更复杂点, 但其实原理都一样
                len_father = len(father)
                # 随机选择两个交叉的点位, while loc1 >= loc2是为了保证两个交叉的点位不会重合并且点位的顺序不会出问题
                loc1, loc2 = 0, 0
                while loc1 >= loc2:
                    loc1 = randint(1, len_father - 1)
                    loc2 = randint(1, len_father - 1)
                # 染色体交叉
                temp = father[loc1:loc2]
                father = father[:loc1] + mother[loc1:loc2] + father[loc2:]
                mother = mother[:loc1] + temp + mother[loc2:]
            # 添加到交叉后的子代种群中
            crossed_population.append(father)
            crossed_population.append(mother)
    return crossed_population


def crossover_pmx(population, pc=0.85):
    """
    部分匹配交叉 partial-mapped crossover, PMX
    PMX保证了每个染色体中的基因仅出现一次, 通过该交叉策略在一个染色体中不会出现重复的基因, 所以通常用于TSP或者其他排序问题中
    PMX类似于两点交叉, 通过随机选择两个交叉点确定交叉区域.
    执行交叉后一般会得到两个无效的染色体, 个别基因会出现重复的情况, 为了修复染色体,
    可以在交叉区域内建立每个染色体的匹配关系, 然后在交叉区域外对重复基因应用此匹配关系消除冲突.

    References: [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling salesman problem", 1985.

    step1 随机选择一对染色体(父代)中几个基因的起止位置(两染色体被选位置相同)
    step2 交换这两组基因的位置, 即做双点交叉
    step3 做冲突检测, 根据交换的两组基因建立一个映射关系, 通过映射关系将发生冲突的基因进行映射, 保证形成的新一对子代基因无冲突

    Args:
        population: 种群
        pc: 交叉概率

    Returns:

    """
    # 经过双点交叉后的新种群, 即 子代
    crossed_population = []
    # 如果种群规模是奇数, 就将最后一个pop出来直接给下一代
    if len(population) % 2:
        crossed_population.append(population.pop())
    else:
        # 步长为2的原因: 生成父代和母代个体用于互相交换染色体
        for i in range(0, len(population), 2):
            father = population[i]
            mother = population[i + 1]
            # 建立基因和索引的映射关系
            pos_father = {value: idx for idx, value in enumerate(father)}
            pos_mother = {value: idx for idx, value in enumerate(mother)}
            # 若生成的随机数小于交叉概率, 则可以发生交叉, 其实就是模拟父代和母代的交配过程
            if random() < pc:
                # 染色体长度, 这儿只针对单染色体的情形, 双/多染色体会更复杂点, 但其实原理都一样
                len_father = len(father)
                # 随机选择两个交叉的点位, while loc1 >= loc2是为了保证两个交叉的点位不会重合并且点位的顺序不会出问题
                loc1, loc2 = 0, 0
                while loc1 >= loc2:
                    loc1 = randint(1, len_father - 1)
                    loc2 = randint(1, len_father - 1)
                # 取出发生交叉的片段
                temp = father[loc1:loc2]
                # 遍历这段染色体里的每个基因
                for gen_idx in range(len(temp)):
                    # 取出交叉片段染色体的每个基因的索引对应的父代和母代的基因val1 val2
                    val1, val2 = father[gen_idx], mother[gen_idx]
                    # 找出在母代染色体上的这个基因在父代染色体上的索引, 和, 在父代染色体上的这个基因在母代染色体上的索引
                    pos1, pos2 = pos_father[val2], pos_mother[val1]
                    # 交换基因
                    father[gen_idx], father[pos1] = father[pos1], father[gen_idx]
                    mother[gen_idx], mother[pos2] = mother[pos2], mother[gen_idx]
                    # 修正基因和索引的映射关系
                    pos_father[val1], pos_father[val2] = pos1, gen_idx
                    pos_mother[val1], pos_mother[val2] = gen_idx, pos2
            # 添加到交叉后的子代种群中
            crossed_population.append(father)
            crossed_population.append(mother)
    return crossed_population


def crossover_sub_tour_exchange(population):
    """
    子路径交换交叉算子

    Args:
        population:

    Returns:

    """
    return population


def crossover_cycle(population):
    """
    循环交叉算子

    Args:
        population:

    Returns:

    """
    return population
