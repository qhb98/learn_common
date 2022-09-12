# coding: utf-8
# @FileName: 2.py
# @Time: 2022/9/12 14:33
# @Author: QHB

"""
使用gurobi 求解Dual LMP问题

"""

from gurobipy import *
model = Model('Dual LMP')
y = {}
for i in range(1, 3):
    y[i] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name = 'y_' + str(i))

model.setObjective(20 * y[1] + 15 * y[2], GRB.MAXIMIZE)

model.addConstr(3*y[1] + 0*y[2] <= 1)
model.addConstr(0*y[1] + 2*y[2] <= 1)
model.addConstr(1*y[1] + 1*y[2] <= 1)
model.addConstr(2*y[1] + 0*y[2] <= 1)
model.addConstr(1*y[1] + 0*y[2] <= 1)
model.addConstr(0*y[1] + 1*y[2] <= 1)

model.optimize()

for key in y.keys():
    print('y[{}] = {}'.format(key, y[key].x))