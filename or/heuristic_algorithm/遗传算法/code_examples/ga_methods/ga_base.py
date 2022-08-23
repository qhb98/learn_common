# coding: utf-8
# @FileName: :ga_base.py
# @Time: 2022/8/8 10:01
# @Author: QHB
import numpy as np
import crossover
import mutation
import ranking
import selection


def func_trans(func):
    pass


class GeneticAlgorithm:
    """
    genetic algorithm 遗传算法

    Parameters
    ----------------
    func : function
        期望优化的目标函数
    n_dim : int
        目标函数中变量的个数
    lb : array_like
        每一个变量的下限
    ub : array_like
        每一个变量的上限
    constraint_eq :
        == 的约束
    constraint_ueq :
        !=  的约束
    precision : array_like

    size_pop : int
        种群大小
    max_iter : int
        最大迭代次数
    prob_mut : float between 0 and 1
        变异概率
    """

    def __int__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001,
                lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(),
                precision=1e-7, early_stop=None):
        self.func = func_trans(func)
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut
        self.n_dim = n_dim

        self.early_stop = early_stop

        # 约束
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)
        self.constraint_ueq = list(constraint_ueq)
        # 变量的下界 shape =
        self.lb = np.array(lb) * np.ones(self.n_dim)
        # 变量的上界 shape =
        self.ub = np.array(ub) * np.ones(self.n_dim)
        # 精度
        self.precision = np.array(precision) * np.ones(self.n_dim)
        lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.lind = np.ceil(lind_raw).astype(int)

        self.int_mode = (self.precision % 1 == 0) & (lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode, self.lb + (np.exp2(self.lind) - 1) * self.precision, self.ub)
        self.len_chrom = sum(self.lind)

        self.Chrom = None
        self.X = None
        self.Y_raw = None
        self.Y = None
        self.FitV = None

        #
        self.generation_best_X = []
        self.generation_best_Y = []

        #
        self.all_history_Y = []
        self.all_history_FitV = []

        #
        self.best_x = None
        self.best_y = None

        # 方法
        self.crtbp()

    def crtbp(self):
        # 创建种群
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):
        # gray码到实数: 一整个染色体
        # 输入是一个 0 和 1 的二维 numpy 数组
        # 输出是一个一维 numpy 数组, 它将每一行输入转换为一个实数
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation
