## 局部搜索算法 local search

### 一、基本概念
局部搜索算法从一个初始解开始, 通过邻域动作, 产生其邻居解, 判断邻居解的质量, 根据某种策略, 
来选择邻居解, 重复上述过程, 至到达终止条件.
不同局部搜索算法的区别就在于: 邻域动作的定义和选择邻居解的策略, 
也是决定算法好坏的关键(集中性和发散性, Intensification and Diversification)

局部搜索算法是解决最优化问题的一种启发式算法, 从爬山算法改进而来, 其基本思想就是
在搜索过程中, 始终选择当前点的邻居中与离目标最近者的方向搜索.

邻域: 在组合优化问题中, 邻域一般定义为由给定转化规则对给定的问题域上每结点进行转化所得到的问题域上结点的集合.
邻域动作: 是一个函数, 通过这个函数, 对当前解s, 产生其相应的邻居解集合. 例如: 对于一个bool型问题, 
    其当前解为s=1001, 当将邻域动作定义为翻转其中一个bit时, 得到的邻居解的集合N(s)={0001, 1101, 1011, 1000}, 
    其中N(s)∈S. 同理, 当将邻域动作定义为互换相邻bit时, 得到的邻居解的集合为N(s)={0101, 1001, 1010}

对于ls算法的改进, 一般从3个角度入手:

    + 局部最优问题  --  考虑到算法在搜索过程中陷入到局部极值点而结束的情况
    + 步长问题  --  步长固定，所以能够搜索到的也仅为一些固定的点, 解决此问题可以在搜索的过程中改变步长(本质为改变映射函数N)
    + 起始点问题  --  随机的选择一些可能解, 分别以这些可能解为初始解进行搜索


### 二、算法的基本流程

    step1 随机选择一个初始的可能解x, 计算P=N(x)为x在映射N下的邻域
    step2 如果满足结束条件, 则go to step9, 其中结束条件包括规定的循环次数、P为空
    step3 begin
    step4 选择P的一个子集, y为此子集的一个最优解
    step5 如果f(y)优于f(x), 则将y改为临时的最优解, 并将P=N(y), 其中f为一指标函数
    step6 否则, 从P中减去刚才选择的子集
    step7 go to step2
    step8 end
    step9 输出计算结果

### 三、通常的案例
TSP问题  
约束机器排序问题  
指派问题 
0-1背包问题  
装箱问题  
SAT问题  
N皇后问题  



