# coding: utf-8
# @FileName: main.py
# @Time: 2022/7/16 17:19
# @Author: QHB

# 导包
import cplex

# 类实例化对象
xiaoming = cplex.Cplex()
xiaoming.read('ex.mps')
xiaoming.solve()

xiaoming.solution.get_objective_value()

xiaoming.solution.get_values()
