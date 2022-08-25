# coding: utf-8
# @FileName: encoder.py
# @Time: 2022/8/24 19:40
# @Author: QHB

# ================================= 实数编码 ========================================== #
import numpy as np


def num2str(num, dec_num=3):
    """
    将浮点数变成string
    Args:
        num: 将数字转换为实数
        dec_num: 浮点数的精度, 默认小数点后3位

    Returns:

    """
    # return str(int(num)) + str(int(num - int(num)) * 1e3)
    s = str(num).replace('.', '')
    s += '0' * abs(int(num) // 10 + 1 + dec_num - len(s))
    return s


def encoder_real_number(x, y):
    """
    实数编码, 将x y两个决策变量的值转换为字符串后拼接在一起组成染色体, 一条染色体即为一个个体
    Args:
        x: 决策变量x的值
        y: 决策变量y的值

    Returns:

    """
    chr_list = []
    for i in range(len(x)):
        chr = num2str(x[i]) + num2str(y[i])
        chr_list.append(chr)
    return chr_list


# ================================= 二进制编码 ========================================== #

def encode_binary(VAR_NUM=2, LB=[-3, -2], UB=[3, 2], EPS=0.01):
    """
    二进制编码
    主要适用于函数求极值的问题
    Args:
        VAR_NUM: 决策变量的数量
        LB: 下区间
        UB: 上区间
        EPS: 编码精度

    Returns:

    """
    # 确定染色体的长度
    L = np.zeros(3)
    # 对应解码时索引从1开始, 此处需要注意, 这样编码可以避免溢出
    L[0] = 0
    # 计算每个决策变量由十进制转换为二进制需要占用的编码数 n
    for i in range(VAR_NUM):
        # (UB - LB) / (2^n - 1) <= EPS ceil向上取整
        L[i + 1] = np.ceil(np.log2((UB[i] - LB[i]) / EPS + 1))
    # 计算染色体的长度
    len_L = int(np.sum(L))
    # 初始化种群 01随机一维数组
    pop = np.random.randint(0, 2, size=len_L)
    # 返回编码好的染色体
    return pop
