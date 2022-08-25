# coding: utf-8
# @FileName: decoder.py
# @Time: 2022/8/24 19:41
# @Author: QHB

# ================================= 实数编码 ========================================== #
import numpy as np


def str2num(s, dec_num=3):
    """
    解码
    Args:
        s: 字符串转换为数值
        dec_num: 浮点数的精度, 默认小数点后3位

    Returns:

    """
    num = int(s[:-dec_num]) + float(s[-dec_num:]) / 10 ** dec_num
    return round(num, dec_num)


def decoder_real_number(chr):
    """
    解码
    Args:
        chr:

    Returns:

    """
    cut = int(len(chr[0]) / 2)
    x = [str2num(chr[i][:cut]) for i in range(len(chr))]
    y = [str2num(chr[i][cut:]) for i in range(len(chr))]
    return x, y


# ================================= 二进制编码 ========================================== #

def decode_binary(bin_list, lb, ub):
    """
    二进制数转换为十进制数
    Args:
        bin_list: 二进制数组
        lb: 下区间
        ub: 上区间

    Returns:

    """
    # 染色体的长度
    length = len(bin_list)
    # 将二进制数转换为十进制数
    temp = np.zeros(length)
    # 索引从1开始, 是因为前面编码的时候染色体的第一个数值为了避免最高位数值溢出默认置0
    for j in range(1, length):
        temp[j] = np.power(2, length - 1 - j) * bin_list[j]
    temp = np.sum(temp)
    # 将十进制数变换到对应的区间
    real = lb + temp * (ub - lb) / (np.power(2, length) - 1)
    return real
