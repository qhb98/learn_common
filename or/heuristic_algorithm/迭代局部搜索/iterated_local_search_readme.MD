
## 迭代局部搜索算法

### 一、基本概念
迭代局部搜索属于探索性局部搜索方法(exploration local search methods)中的一种, 
在局部搜索得到的局部最优解的基础上加入了扰动, 然后再重新进行局部搜索.  



### 二、算法流程

    step1 初始状态: best solution, current solution
    step2 从初始解中进行局部搜索, 找到一个局部最优解s1
    step3 扰动s1, 获得新的解s2
    step4 从新解s2中进行局部搜索, 再次找到一个局部最优解s3
    step5 基于判断策略, 对s3好坏进行判断, 选择是否接受s3作为新的best solution
    step6 直到达到边界条件, 不然跳回第二步一直循环搜索.




