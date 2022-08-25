from random import random

import numpy as np

"""

�Ŵ��㷨�г�����ѡ������

"""


def selection_roulette(population, population_fitness):
    """
    ���̶�ѡ������
    Args:
        population: ��Ⱥ
        population_fitness: ��Ⱥ��ÿ�������Ӧ����Ӧ��ֵ��ɵļ���

    Returns: ���ؾ������̶�ѡ��������Ⱥ

    """
    # ��Ⱥ��Ӧ��֮��
    sum_fitness = sum(population_fitness)
    prob = [population_fitness[i] / sum_fitness for i in range(len(population))]
    chosen = []
    # ���̶ĵ�ѡ�����Ӧ������Ⱥ��ģһ��, ��һ�α�����Ϊ�˱�֤ѡ��ǰ��ĸ������Ӵ�֮�����Ⱥ��ģ����ȫһ�µ�
    for i in range(len(population)):
        cum = 0
        # �������
        m = random()
        # ���̶�ѡ��
        for j in range(len(population)):
            # �����ۼ�
            cum += prob[j]
            if cum >= m:
                chosen.append(population[j])
                break
    return chosen


def selection_tournament(tour_size=3, population, population_fitness):
    """
    ������ѡ������
    �ڽ�����ѡ�񷽷���ÿһ���У������������ѡ�������������壬������Ӧ�ȵ÷���ߵĻ�ʤ����ѡ��

    Args:
        tour_size: ÿ��ѡ��ĸ�������
        population: ��Ⱥ
        population_fitness: ��Ⱥ��ÿ���������Ӧ��ֵ��ɵļ���

    Returns:

    """
    # ����ѡ��ĸ�����ԭ��Ⱥ�е�����
    select_idx = []
    for i in range(len(population)):
        # aspirants_index = np.random.choice(range(len(population)), size=tour_size)
        # ����Ⱥ�������ȡtour_size������, randint������ʵ������, random.choice������TSP����
        aspirants_index = np.random.randint(range(len(population)), size=tour_size)
        select_idx.append(max(aspirants_index, key=lambda i: population_fitness[i]))
    # ��������������Ⱥ, �õ���һ����Ⱥ
    population = population[select_idx, :]
    return population


def selection_tournament_faster(tour_size=3, population, population_fitness):
    """
    ��selection_tournamentһ��, ֻ�����������ϼ����� numpy������
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
    ����������� SUS
    ���������������ǰ����������ѡ����޸İ汾
    ʹ����ͬ�����̣�������ͬ����ʹ�ö��ѡ��㣬ֻ��תһ��ת�̾Ϳ���ͬʱѡ�����и���

    step1 ����ָ��ļ�� p = sum(fitness) / len(population)
    step2 ����������ָ��λ�� start = np.random.randint(p)
    step3 �����ָ���λ�� pointers = [start + i * p], i=0,1,...,len(population) - 1
    step4 ���ݸ���ָ���λ��ѡ���N������

    Args:
        population: ��Ⱥ
        population_fitness: ��Ⱥ��ÿ�������Ӧ����Ӧ��ֵ��ɵļ���

    Returns:

    """
    # ��Ⱥ��Ӧ��֮��
    sum_fitness = sum(population_fitness)
    # prob = [population_fitness[i] / sum_fitness for i in range(len(population))]
    # prob_cum = [sum(prob[:i+1]) for i in range(len(prob))]
    chosen = []
    # ����ָ��ļ��
    p = sum_fitness / len(population)
    for i in range(len(population)):
        pos_cur = np.random.randint(p, size=1) + i * p
        chosen.append(population[pos_cur, :])
    return chosen
