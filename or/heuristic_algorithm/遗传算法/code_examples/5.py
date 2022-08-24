# coding: utf-8
# @FileName: 5.py
# @Time: 2022/8/24 16:58
# @Author: QHB
"""
example_5:
遗传算法求解TSP问题

旅行商问题：给定一系列城市和每对城市之间的距离，求解访问每一座城市一次并回到起始城市的最短回路

提供的cities.txt 是 127 个城市的 x 和 y 坐标数据

基本思路:
1. 要求的是最佳的路径, 路径的结果可以直接用城市的ID的顺序表征. 因此, 可以将路径的结果视作个体
2. 种群就是一堆的路径结果组成的集合, 每一代里的最佳个体就是当代的种群里路径结果最好的那条路径
3. 那么接下来只要对路径做选择 交叉 变异, 循环迭代, 就可以了
4. 主要需要考虑编码, 即如何将路径编码, 因为是用城市的ID的顺序表征路径, 那最佳的方式肯定就是实数编码

"""
import random
from typing import List

import numpy as np
import math
import time


def load_data(data_path) -> List[tuple]:
    """
    导入数据，得到城市坐标信息
    Args:
        data_path: 数据文件地址 str

    Returns: 所有城市的坐标信息 二维 list

    """
    cities = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x_str, y_str = line.split()[1:]
            x, y = int(x_str), int(y_str)
            cities.append((x, y))
    return cities


def get_cities_distance(cities):
    """
    计算城市两两之间的距离
    Args:
        cities: 所有城市的二维坐标

    Returns: 城市的相对距离矩阵

    """
    city_num = len(cities)
    dis_matrix = np.zeros((city_num, city_num))
    for i in range(city_num - 1):
        for j in range(i + 1, city_num):
            dist = get_two_cities_dist(cities[i], cities[j])
            dis_matrix[i, j] = dist
            dis_matrix[j, i] = dist
    return dis_matrix


def get_two_cities_dist(city1, city2):
    """
    计算两个城市之间的距离
    Args:
        city1:
        city2:

    Returns:

    """
    x_1, y_1 = city1
    x_2, y_2 = city2
    return math.sqrt(math.pow(x_1 - x_2, 2) + math.pow(y_1 - y_2, 2))


def get_route_fitness_value(route, dist_matrix):
    """
    计算某一路线的适应度, 适应度值就是 1/(整条路线的总距离+最后一个城市到第一个城市的距离)
    路线距离总和越长, 则适应度值就越小
    (因为要求要回到最初点, 如果没有这个要求, 就不需要+最后一个城市到第一个城市的距离 了)
    Args:
        route: 当前的路线
        dist_matrix: 距离矩阵

    Returns: 当前路线的适应度值

    """
    dist_sum = 0
    for i in range(len(route) - 1):
        dist_sum += dist_matrix[route[i], route[i + 1]]
    dist_sum += dist_matrix[route[len(route) - 1], route[0]]
    return 1 / dist_sum


def get_all_routes_fitness_value(routes, dist_matrix):
    """
    计算所有路线的适应度
    Args:
        routes: 所有路线
        dist_matrix: 距离矩阵

    Returns: 所有路线的适应度

    """
    fitness_values = np.zeros(len(routes))
    for i in range(len(routes)):
        f_value = get_route_fitness_value(routes[i], dist_matrix)
        fitness_values[i] = f_value
    return fitness_values


def init_route(n_route, n_cities):
    """
    通过随机采样的方式生成初始化的路线, 循环迭代后得到初始化的种群
    Args:
        n_route: 初始化的路线数量, 即初始化的种群的规模
        n_cities: 城市的数量

    Returns: 初始化的种群, 以二维矩阵的形式展现, 每一行表示种群中的一个个体, 每一列表示这个个体的基因, 每个个体只有一条染色体

    """
    routes = np.zeros((n_route, n_cities)).astype(int)
    for i in range(n_route):
        # 从n_cities中随机采样n_cities个数, 组成初始的路线
        routes[i] = np.random.choice(range(n_cities), size=n_cities, replace=False)
    return routes


def selection(routes, fitness_values):
    """
    选择算子
    Args:
        routes: 所有的路线
        fitness_values: 所有路线的适应度值

    Returns: 选择后的所有路线

    """
    selected_routes = np.zeros(routes.shape).astype(int)
    n_routes = routes.shape[0]
    # 纯粹的按概率选择
    # probability = fitness_values / np.sum(fitness_values)
    # for i in range(n_routes):
    #     choice = np.random.choice(range(n_routes), p=probability)
    #     selected_routes[i] = routes[choice]
    # 轮盘赌选择, 对比按概率选择可以看到收敛速度明显加快
    probability = fitness_values / np.sum(fitness_values)
    roulette_probability = 0
    for p in range(len(probability)):
        roulette_probability += probability[p]
        probability[p] = roulette_probability
    for i in range(n_routes):
        random_m = random.uniform(0, 1)
        for j in range(len(probability)):
            if random_m < probability[j]:
                selected_routes[i] = routes[j]
                break
    return selected_routes


def crossover(routes, n_cities):
    """
    单点交叉算子
    Args:
        routes: 所有路线
        n_cities: 城市数量

    Returns: 交叉选择后的所有路线

    """
    for i in range(0, len(routes), 2):
        r1_new, r2_new = np.zeros(n_cities), np.zeros(n_cities)
        seg_point = np.random.randint(0, n_cities)
        cross_len = n_cities - seg_point
        r1, r2 = routes[i], routes[i + 1]
        r1_cross, r2_cross = r2[seg_point:], r1[seg_point:]
        r1_non_cross = r1[np.in1d(r1, r1_cross) == False]
        r2_non_cross = r2[np.in1d(r2, r2_cross) == False]
        r1_new[:cross_len], r2_new[:cross_len] = r1_cross, r2_cross
        r1_new[cross_len:], r2_new[cross_len:] = r1_non_cross, r2_non_cross
        routes[i], routes[i + 1] = r1_new, r2_new
    return routes


def mutation(routes, n_cities, prob=0.01):
    """
    基本位变异算子
    Args:
        routes:
        n_cities:

    Returns:

    """
    # 产生一组[0, 1)的随机样本, 样本数量为len(routes)
    p_rand = np.random.rand(len(routes))
    for i in range(len(routes)):
        if p_rand[i] < prob:
            mut_position = np.random.choice(range(n_cities), size=2, replace=False)
            l, r = mut_position[0], mut_position[1]
            routes[i, l], routes[i, r] = routes[i, r], routes[i, l]
    return routes


if __name__ == '__main__':
    # 记录程序开始运行的时间
    start = time.time()
    # 路线的数量, 其实就是种群的规模, 一条路线表示一个个体
    n_routes = 100
    # 最大迭代次数
    epoch = 10000
    # 导入数据
    cities = load_data("data/cities.txt")
    # 计算城市距离矩阵
    dist_matrix = get_cities_distance(cities)
    # 初始化所有路线
    routes = init_route(n_routes, len(cities))
    # 计算所有初始路线的适应度
    fitness_values = get_all_routes_fitness_value(routes, dist_matrix)
    # 返回适应度值最高的路线所对应的在种群中的索引
    best_index = fitness_values.argmax()
    # 保存最优路线及其适应度
    best_route, best_fitness = routes[best_index], fitness_values[best_index]

    # 记录解没有发生变化的次数, 若提早陷入最优解(局部最优解/全局最优解),
    not_improve_time = 0
    # 循环迭代
    for i in range(epoch):
        # 选择
        routes = selection(routes, fitness_values)
        # 交叉
        routes = crossover(routes, len(cities))
        # 变异
        routes = mutation(routes, len(cities), prob=0.01)
        # 计算当代种群的适应度值
        fitness_values = get_all_routes_fitness_value(routes, dist_matrix)
        # 返回最佳个体在当代种群中的索引
        best_route_index = fitness_values.argmax()
        if fitness_values[best_route_index] > best_fitness:
            not_improve_time = 0
            best_route, best_fitness = routes[best_route_index], fitness_values[best_route_index]  # 保存最优路线及其适应度
        else:
            not_improve_time += 1
        if (i + 1) % 200 == 0:
            print('epoch: {}, 当前最优路线距离： {}'.format(i + 1, 1 / get_route_fitness_value(best_route, dist_matrix)))
        if not_improve_time >= 2000:
            print('连续2000次迭代都没有改变最优路线，结束迭代')
            break

    print('最优路线为：')
    print(best_route)
    print('总距离为： {}'.format(1 / get_route_fitness_value(best_route, dist_matrix)))

    end = time.time()
    print('耗时: {}s'.format(end - start))
