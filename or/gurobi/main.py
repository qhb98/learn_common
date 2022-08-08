# coding: utf-8
# @FileName: :main.py
# @Time: 2022/8/5 21:45
# @Author: QHB
"""
使用gurobi解决VRP问题, python API接口

"""
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

rnd = np.random
rnd.seed(0)

# number of clients
n = 10

xc = rnd.rand(n + 1) * 200
yc = rnd.rand(n + 1) * 100
# cplex_figure out
plt.plot(xc[0], yc[0], c='r')
plt.scatter(xc[1:], yc[1:], c='b')
plt.show()

N = [i for i in range(1, n + 1)]
V = [0] + N
A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j]) for i, j in A}
Q = 20
q = {i: rnd.randint(1, 10) for i in N}

mdl = Model('CVRP')
x = mdl.addVars(A, vtype=GRB.BINARY)
u = mdl.addVars(N, vtype=GRB.CONTINUOUS)
mdl.modelSense = GRB.MAXIMIZE
mdl.setObjective(quicksum(x[i, j] * c[i, j] for i, j in A))
mdl.addConstr(quicksum(x[i, j] for j in V if j != i) == 1 for i in N);
mdl.addConstr(quicksum(x[i, j] for i in V if i != j) == 1 for j in N);
mdl.addConstr((x[i,j] == 1) >> (u[i] + q[i] == u[j]) for i,j in A if i != 0 and j != 0);
mdl.addConstr(u[i] >= q[i] for i in N);
mdl.addConstr(u[i] >= Q for i in N);
mdl.optimize()
