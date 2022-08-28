
## 蚁群算法 ant clony optimization, ACO


### 一、基本概念

ACO是一种群智能算法, 由一群无智能或有轻微智能的个体agent通过相互协作而表现出智能行为, 从而求解复杂问题.  

最早用于求解TSP问题, 并且表现出了很大的优越性, 因为分布式特性、鲁棒性强, 而且容易与其他算法结合, 但是同时也
存在着收敛速度慢、容易陷入局部最优等缺点.     

原理: 假如蚁群中所有蚂蚁的数量为m, 所有城市之间的信息素用矩阵pheromone表示, 用蚂蚁的行走路径表示待优化问题的可行解,
整个蚂蚁群体的所有路径构成待优化问题的解空间, 路径较短的蚂蚁释放的信息素量较多.  

最短路径为best_length, 最佳路径为best_tour. 每只蚂蚁都有一个禁忌表tabu来存储该蚂蚁已经访问过的城市,  
表示其在以后的搜索中将不能访问的城市列表; 还有用另一个允许访问的城市表allowed来存储还可以访问的城市列表;  
另外还用矩阵delta来存储它在一个循环/迭代中给所经过的路径释放的信息素等.  


### 二、算法原理

假设整个蚂蚁群体中蚂蚁的数量为m, 城市的数量为n, 城市i和城市j之间的距离为d_ij, t时刻城市i和城市j连接路径上
的信息素浓度为theta_ij(t). 

初始时刻, 各个城市间连接路径上的信息素浓度相同, 均为theta_ij(0) = theta_0.  

蚂蚁k根据各个城市间连接路径上的信息素浓度决定下一个访问城市, 设P_k_ij(t)表示t时刻蚂蚁k从城市i转移到城市j的转移概率,
公式为:  

    P_k_ij = ([theta_ij(t)]^aif * [b_ij(t)]^beta) / sum_for all s([theta_is(t)]^aif * [b_is(t)]^beta), if s ∈ allow_k  
    P_k_ij = 0, s not ∈ allow_k


b_ij(t) = 1 / d_ij  表示启发函数, 表示蚂蚁从城市i转移到城市j的期望程度  
allow_k = 表示蚂蚁k待访问城市的集合, when start, allow_k 中有 len(cities) - 1个元素  
aif 表示信息素重要程度因子, 值越大, 表示信息素的浓度在转移中起到的作用越大, 若aif=0, 就变成了传统的贪心算法  
beta 表示启发函数重要程度因子, 值越大, 表示启发函数在转移中的作用越大, 即蚂蚁会以较大的概率转移到距离短的城市, 
若beta=0, 就成了纯粹的正反馈的启发式算法    


设参数 0 < rou < 1, 表示信息素的挥发程度. 因此, 当所有蚂蚁完成一次循环后, 各个城市间连接路径上的信息素浓度
需要进行实时更新, 即  

    theta_ij(t+1) = (1 - rou) theta_ij(t) + derta theta_ij
    derta theta_ij = sum_k(derta theta_k_ij)

derta theta_k_ij 表示第k只蚂蚁在城市i与城市j连接路径上连接路径上释放的信息素浓度  
derta theta_ij 表示所有蚂蚁在城市i与城市j连接路径上释放的信息素浓度之和  


算法流程:
    
    step1 初始化参数
    step2 构建解空间  将各个蚂蚁随机地置于不同的出发点, 对每个蚂蚁按照转移概率计算公式确定下一个待访问的城市, 直到所有蚂蚁访问完所有城市,
            即构造完一组路径
    step3 更新信息素  计算各个蚂蚁经过的路径长度, 根据信息素迭代公式对各个城市路径上的信息素浓度进行更新, 同时
            记录当前迭代次数中的最优解
    step4 判断是否终止



### 三、针对蚂蚁信息素释放的问题提出的三个模型

1.ant cycle system(常选)   
利用蚂蚁经过路径的整体信息(经过路径的总长)计算释放的信息素浓度  

    derta theta_k_ij = Q / L_k, 第k只蚂蚁从城市i访问城市j
    derta theta_k_ij = 0, other

Q为常数, 表示蚂蚁循环一次所释放的信息素总量  
L_k为第k只蚂蚁经过路径的长度  


2.ant quantity system  
利用蚂蚁经过路径的局部信息(经过各个城市间的距离)计算释放的信息素浓度  

    derta theta_k_ij = Q / d_ij, 第k只蚂蚁从城市i访问城市j
    derta theta_k_ij = 0, other


3.ant density system  
将信息素释放的浓度设置为固定值, 并没有考虑到不同蚂蚁经过路径长度的影响  

    derta theta_k_ij = Q, 第k只蚂蚁从城市i访问城市j
    derta theta_k_ij = 0, other




### 参考资料
https://mp.weixin.qq.com/s/poQT_V46ZeucCfx0ctIPvQ
