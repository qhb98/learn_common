
## 分支定界 branch and bound

所谓分支定界法就是缩小上界, 增大下界来得到最优目标函数(通俗点说就是给对应整数规划问题A的松弛线性规划问题B不断添加约束条件,
直到有符合整数条件的最优解).

主要分为三个步骤:
1. 分支
   从原LP问题中划分出两个子问题, 这两个子问题的可行域都是原线性规划问题的子集.
   每一次分支得到的子问题最优解的目标函数值, 都小于等于分之前的最优解的目标函数值. 非整数解的最大值作为新的上界.
   
    
2. 定界



3. 比较和剪枝




用于评估分支定界算法表现的因素主要有 求解算法的运行时间 以及 被探索完毕的节点数量


### 分支定界算法的策略
搜索策略（如何选择下一个被探索的节点） 
   
   depth-first search 深度优先搜索:
      优点 - 占用内存小, 可重复使用父节点的信息
      缺点 - 单一分支约束的存在会导致不可行性trashing, unbounded depth
      改进 - iterative deepening DFS设置探索的路径深度上限, 
            interleaved depth-first search 按顺序从每个不同的DFS路径中选择一个子问题进行探索
            depth-first search with complete branching 计算出所有子问题的下界
   
   breadth-first search 广度优先搜索
      优点 - BrFS在探索任何更深层的子问题之前, 先探索与根保持固定距离的所有子问题. BrFS策略的优点是总能找到最接近树根的最优解
      缺点 - 由于最优解常位于更大的深度, BrFS通常不能利用bound剪枝

   best-first search 最佳优先搜索
      优点 - 使用最佳函数 measure of best 

   cyclic best-first search 循环最佳优先搜索:
      CBFS结合BFS和DFS，将未开发的子问题（BFS的堆结构存储）划分为若干个contour


分枝策略（影响子节点的数量和子问题如何被分解）


剪枝策略（决定节点是否被探索完毕）
初始解的选择（可使用启发式算法得到）
并行计算（需要关注不同处理器间的信息交换，包括bound/dominance/pruning rule） 
我们将分支定界算法分为两个阶段，搜索阶段和验证阶段。其中，搜索阶段旨在找到当前分支的最优节点（incumbent solution），
涉及搜索策略和分支策略，如切平面作为剪枝策略可用于搜索阶段，用于鉴别可行解
而验证阶段旨在验证其他分支被探索完毕（如被剪枝，找到当前节点的最优解等），涉及分支策略和剪枝策略








参考链接:
https://www.cnblogs.com/cruelty_angel/p/10529490.html
https://www.zhihu.com/question/449825799