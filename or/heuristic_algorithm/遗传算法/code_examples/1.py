# coding: utf-8
# @FileName: 1.py
# @Time: 2022/8/23 9:49
# @Author: QHB

"""
example_1:
在一个长度为n的数组nums中选择10个元素, 使得10个元素的和与原数组的所有元素之和的1/10无限接近.
例如:
    min |sum(answer) - sum(nums)|
    s.t.
        n = 50
        sum(nums) = 1000

思路:
    1. 创建一个包含100个解的初始解集
    2. 适应度计算
    3. 交叉
    4. 变异
    5. 循环迭代

少了编解码的环节和选择的环节, 主要目标是理解 交叉 和 变异 这个过程.

"""

import random


# ====================================  初始化种群   ==================================== #
def create_answer(numbers, n, k=10):
    """
    生成k个[0, 1000]的随机数的个体, 组成规模为n的种群
    Args:
        numbers: 随机选取的范围列表
        n: 种群规模
        k: 每个个体的染色体数

    Returns:

    """
    result = []
    for i in range(n):
        # 不改变原列表, 截取列表的指定长度的随机数
        result.append(random.sample(numbers, k))
    return result


# ====================================  计算适应度值   ==================================== #
def error_level(new_answer, number_set):
    """
    根据|sum(answer) - sum(nums)|计算种群中每个个体和目标之间的偏差error
    error越小, 适应度值越大
    Args:
        new_answer: 种群
        number_set: 数组nums

    Returns: 适应度值

    """
    # 存储种群中所有个体的偏差
    error = []
    # 目标值, 数组总和的1/10
    right_answer = sum(number_set) / 10
    for item in new_answer:
        # 根据 |sum(answer) - sum(nums)| 计算偏差
        value = abs(right_answer - sum(item))
        # 根据偏差计算适应度值
        if value == 0:
            error.append(10)
        else:
            error.append(1 / value)
    return error


# ====================================  变异   ==================================== #
def variation(old_answer, number_set, pro):
    """
    对种群进行变异, 生成新种群
    Args:
        old_answer: 原种群
        number_set: 数组nums
        pro: 变异概率

    Returns:

    """
    for i in range(len(old_answer)):
        rand = random.uniform(0, 1)
        if rand < pro:
            # 采用基本位变异的方法
            rand_num = random.randint(0, 9)
            old_answer[i] = old_answer[i][:rand_num] + random.sample(number_set, 1) + old_answer[i][rand_num + 1:]
    return old_answer


# ====================================  交叉选择生成新的种群   ==================================== #
def choice_selected(old_answer, number_set):
    """
    根据原种群生成新的种群, 采用两点交叉的方法
    Args:
        old_answer: 原种群
        number_set: 原数组nums

    Returns:

    """
    result = []
    # 采用轮盘赌法确定选择概率
    error = error_level(old_answer, number_set)
    error_one = [item / sum(error) for item in error]
    for i in range(1, len(error_one)):
        error_one[i] += error_one[i - 1]
    # 将种群均分成父体和母体
    for i in range(len(old_answer) // 2):
        temp = []
        # 选择 父体和母体
        for j in range(2):
            rand = random.uniform(0, 1)
            for k in range(len(error_one)):
                if k == 0:
                    if rand < error_one[k]:
                        temp.append(old_answer[k])
                else:
                    if error_one[k - 1] <= rand < error_one[k]:
                        temp.append(old_answer[k])
        # 两点交叉, 父体和母体交叉, 这里的(0, 6)以及3都跟每个个体的规模为10有直接关系
        rand = random.randint(0, 6)
        temp_1 = temp[0][:rand] + temp[1][rand:rand + 3] + temp[0][rand + 3:]
        temp_2 = temp[1][:rand] + temp[0][rand:rand + 3] + temp[1][rand + 3:]
        result.append(temp_1)
        result.append(temp_2)
    return result


# ====================================  主函数   ==================================== #
if __name__ == '__main__':
    # 生成nums数组
    number_set = random.sample(range(0, 1000), 50)
    # 确定种群规模, 随机初始化种群
    initial_answer = create_answer(number_set, n=100, k=10)
    # 初始解
    first_answer = initial_answer[0]
    greatest_answers = []
    # 迭代1000代
    for i in range(1000):
        # 计算种群的适应度
        error = error_level(initial_answer, number_set)
        # 确定适应度值最大的个体在种群中的索引
        index = error.index(max(error))
        # 交叉
        initial_answer = choice_selected(initial_answer, number_set)
        # 变异
        initial_answer = variation(initial_answer, number_set, 0.1)
        # 记录当前的最优个体及适应度值
        greatest_answers.append([initial_answer[index], error[index]])
    # 对每代的最优解进行排序
    greatest_answers.sort(key=lambda x: x[1], reverse=True)
    print("正确答案为", sum(number_set) / 10)
    print("给出最优解为", greatest_answers[0][0])
    print("该和为", sum(greatest_answers[0][0]))
    print("选择系数为", greatest_answers[0][1])
    print("最初解的和为", sum(first_answer))
