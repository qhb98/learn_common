### cplex 下载 安装 使用 学习过程记录

### 1. 下载安装参考链接： 

    https://blog.csdn.net/cc098818/article/details/99619928
    https://blog.csdn.net/wuxiaolongah/article/details/120339847


### 2. 入门学习参考链接： 

    https://www.bilibili.com/video/BV1Kb411E7tT?spm_id_from=333.337.search-card.all.click&vd_source=7111d4cfa9354342c253c06ecdd64e2f



![默认安装位置](/or/cplex/cplex_figure/cplex_默认安装位置.png)


![界面截图](/or/cplex/cplex_figure/cplex_界面截图.png)


### 3. cplex 组件介绍

+ opl 是建模环境和集成开发环境IDE
+ cplex 是优化器, 包括简单型、障碍型和混合整数优化器
+ cpoptimizer 是CP优化器, 可以和优化器配合使用
+ python 建模层 DOcplex, 用于cplex和CP优化器



### 4. 快速入门cplex

    参考链接： 
    https://www.bilibili.com/video/BV1ot411X79Z?spm_id_from=333.337.search-card.all.click&vd_source=7111d4cfa9354342c253c06ecdd64e2f


### 5. 注意点

项目路径和文件配置都要是英文路径, 否则会报错
    

![配置截图](/or/cplex/cplex_figure/cplex配置截图.png)


### 6. 基本语法

    // dvar 定义变量
    dvar int+ x;
    dvar int+ y;
    dvar boolean x[1..n];
    
    // minimize  和 maximize 定义目标求最大值还是最小值
    minimize 2*x + 3 * y;
    maximize sum(j in 1..n) p[j] * x[j];

    // subject to {  }  定义求解的约束条件
    subject to{
      2 * x + 3 * y >= 20;
      x + y >= 0;
    }
    
    // 定义常量
    int n = 4;
    int p[1..n] = [12, 11, 9, 8];
    
    // 求和符号
    forall(j in 1..5)
    sum(i in 1..5) x[i][j] ==1;

### 7. MPS 和 LP 文件

优化求解器如 cplex 或 gurobi，都支持直接读取线性规划建模文件 MPS 格式或 LP 格式，因此，有必要研究一下，这样就可以编写一个 mps 或 lp 文件，用不同的求解器直接算了。

MPS文件的例子：

    NAME        chen
    ROWS
     N  obj
     L  c1
     L  c2
     E  c3
    COLUMNS
        x1        obj                 -1   c1                  -1
        x1        c2                   1
        x2        obj                 -2   c1                   1
        x2        c2                  -3   c3                   1
        x3        obj                 -3   c1                   1
        x3        c2                   1
        MARK0000  'MARKER'                 'INTORG'
        x4        obj                 -1   c1                  10
        x4        c3                -3.5
        MARK0001  'MARKER'                 'INTEND'
    RHS
        rhs       c1                  20   c2                  30
    
    BOUNDS
     UP bnd       x1                  40
     LO bnd       x4                   2
     UP bnd       x4                   3
    ENDATA

NAME 表示这个优化模型的名字, 后面可以不写内容

ROWS 表示每一行，包括目标函数与约束条件  
    N 表示自由行， obj 是对目标函数的命名，可以任意取名  
    L 表示该行小于等于，c1 是对该行的命名，可以任意取名  
    G 表示该行大于等于  
    E 表示该行等于  

COLUMNS 表示每一列，以及对应的系数  
    下面的第一列要空    
    下面的第二列表示列的名字，其实就是求解变量  
    下面的第三列表示所在行的名字  
    下面的第四列表示所在行与列对应的系数  
    其中 MARK0000 'MARKER' 'INTORG' MARK0001 'MARKER'  'INTEND' 分别表示整数变量的起止  
    第五列、第六列分别与第三列、第四列的含义相同  

RHS: 约束条件最右端的数字
    下面的第一列要空  
    下面的第二列表示 rhs 名字，可以任取  
    下面的第三列表示所在行的名字  
    下面的第四列表示所在行对应的 RHS 值  
    第五列、第六列分别与第三列、第四列的含义相同  

Bounds: 表示各变量的上界或下界  
    LO 表示下界  
    UP 表示上界  
    FX 表示该变量固定值  
    FR 表示改变量的范围为 ( − ∞ , ∞ ) (-\infty,\infty)(−∞,∞)  
    MI 表示下界为负无穷  
    PL 表示上界为正无穷  
    MPS 变量默认的范围为 [ 0 , ∞ )   

ENDDATA: 表示 MPS 文件结束

    
LP文件例子：
    
    Maximize
     obj: x1 + 2 x2 + 3 x3 + x4
    Subject To
     c1: - x1 + x2 + x3 + 10 x4 <= 20
     c2: x1 - 3 x2 + x3 <= 30
     c3: x2 - 3.5 x4 = 0
    Bounds
     0 <= x1 <= 40
     2 <= x4 <= 3
    General
     x4
    End

LP文件非常清晰明了，但貌似规划软件对 mps 文件支持的更好。其中需要注意的是

    Bounds 里面若有 free 表示该变量无上下界
    General 表示整数变量
    Binary 表示二进制变量
    \ 表示注释

cplex 有自己的字段 pwl 表示分段线性约束