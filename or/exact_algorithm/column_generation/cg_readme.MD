
## 列生成 column generation

### 一

列生成是一种用于求解大规模线性优化问题的高效算法, 理论基础上由Danzi等人于1960年提出. 
本质上而言, 列生成算法就是单纯形法的一种形式, 上用于求解线性规划问题的. 
目前已经被应用于求解以下著名的NP-hard优化问题: 机组人员调度问题 crew assignment problem,
切割问题 cutting stock problem, 车辆路径问题 vehicle routing problem, 
单资源工厂选址问题 the single facility location problem 等.

### 二 larger linear programs
在某些线性优化问题的模型中, 约束的数目有限, 但是变量的数目随着问题规模的增长会爆炸式地增长, 
因此不能把所有的变量都显性地在模型中表达出来. 这类问题就是大规模的线性优化问题.

### 三 流程

1. 先把原问题(master problem) restrict(restrict的方法看情况, 可以直接用启发式算法)到一个规模更小
(即变量个数比原问题少)的restricted master problem, 在 restricted master problem 上用单纯形法求最优解, 但是此时求得的最优
解只是restricted master problem上的, 并不是master problem的最优解.
2. 此时, 就需要通过一个subproblem去check那些未被考虑的变量中是否有使得reduced cost小于0的.
如果有, 那么就把这个变量的相关系数列加入到restricted master problem的系数矩阵中, 回到第一步.

经过反复的迭代, 知道subproblem中的reduced cost rate大于等于0, 那么master problem就求到了最优解.









参考链接:  
https://www.cnblogs.com/dengfaheng/p/11249879.html

