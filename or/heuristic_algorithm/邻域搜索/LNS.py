
############## 函数库声明 ###########################
import random #关于随机数的函数库
import xlrd   #读取excel的函数库
from copy import deepcopy  #复制变量的函数
import matplotlib.pyplot as plt #画图

############## 类声明 ###############################
class Node:   #将节点的所有信息打包为Node类变量
    def __init__(self):
        self.node_id = 0 #节点编号
        self.x = 0.0     #节点横坐标
        self.y = 0.0     #节点纵坐标

class vehicle:  #解的所有信息打包为Vehicle类变量
    def __init__(self):
        self.route = []         #遍历节点的顺序
        self.cost = 0.0       #解对应的总成本

############## 数据导入 #############################
#将"input_node.xlsx"中的节点信息汇总为集合N
N = {}
book = xlrd.open_workbook("input_node.xlsx")
sh = book.sheet_by_index(0)
for l in range(1, sh.nrows):  # read each lines
    node_id = str(int(sh.cell_value(l, 0)))
    node = Node()
    node.node_id = int(sh.cell_value(l, 0))
    node.x = float(sh.cell_value(l, 1))
    node.y = float(sh.cell_value(l, 2))
    N[l-1] = node

#根据节点信息计算距离矩阵A，A[i][j]代表从i节点到j节点的出行成本
A = [[0 for j in range(sh.nrows-1)] for i in range(sh.nrows-1)]
for i in range(sh.nrows-1):
    for j in range(sh.nrows-1):
        A[i][j] = ((N[i].x-N[j].x)**2+(N[i].y-N[j].y)**2)**0.5

############## 参数设置 ############################
Iter = 300       #迭代次数
random.seed(1) #随机种子
num_node = sh.nrows-1 #节点数量
num_destroy = int(num_node*0.2) #破坏程度

############## 函数定义 ############################
#计算成本的函数
def get_route_cost(route):
    cost = 0
    for i in range(1, len(route)):
        cost += A[route[i-1]][route[i]]
    return cost

#破坏算子函数
def destroy(solution, num_destroy):
    destroy_node_bank = []
    while len(destroy_node_bank) < num_destroy:
        n = random.randint(0, num_node-1) #随机产生一个需要破坏的节点
        while n in destroy_node_bank:   #若产生了重复的随机数，则重新生成
            n = random.randint(0, num_node-1)
        destroy_node_bank.append(n)     #将破坏的节点放入destroy_node_bank
        solution.route.remove(n)        #在解中删除破坏的节点
    return solution, destroy_node_bank

#修复算子函数
def repair(solution,destroy_node_bank):
    for n in destroy_node_bank:
        #计算将n插入各个位置的成本
        insert_list = [0 for i in range(len(solution.route))]
        for i in range(0, len(solution.route)):
            insert_list[i] = A[solution.route[i-1]][n]+A[n][solution.route[i]]-A[solution.route[i]][solution.route[i-1]]
        #将n插入最优的位置
        greedy_index = insert_list.index(min(insert_list))
        solution.route.insert(greedy_index, n)
    return solution

#画图
def plot_best(solution, count_iter):
    scatter_x = [N[solution.route[i]].x for i in range(num_node)]
    scatter_y = [N[solution.route[i]].y for i in range(num_node)]
    plt.scatter(scatter_x, scatter_y)
    plt_x = scatter_x
    plt_x.append(N[solution.route[0]].x)
    plt_y = scatter_y
    plt_y.append(N[solution.route[0]].y)
    plt.plot(plt_x, plt_y)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylim(-20, 120)
    plt.text(25, -10, '迭代次数='+str(count_iter+1)+'，历史最优解目标值='+str(round(solution.cost,2)))

############## 产生初始解 ############################
solution=vehicle()
solution.route = [i for i in range(num_node)]  #按照节点编号依次相连构成初始解
solution.cost = get_route_cost(solution.route) #计算初始解对应的成本
best_solution = deepcopy(solution)             #初始化历史最优解
best_record = [0 for i in range(Iter)]

for count_iter in range(Iter):
    tem_solution = deepcopy(solution)
    #破坏当前解
    tem_solution, destroy_node_bank=destroy(tem_solution,num_destroy)
    #修复当前解
    neighbor_solution = repair(tem_solution, destroy_node_bank)
    #计算邻域解的目标值
    neighbor_solution.cost = get_route_cost(neighbor_solution.route)
    #判断邻域解是否接受（比历史解要好）
    if neighbor_solution.cost<best_solution.cost:
        solution = deepcopy(neighbor_solution)
        best_solution = deepcopy(neighbor_solution)
    best_record[count_iter] = best_solution.cost

plt.figure()
plt.plot([i+1 for i in range(Iter)], best_record)
plt.xlabel('迭代次数')
plt.ylabel('历史最优值')
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.scatter([i+1 for i in range(Iter)],best_record,)
plt.show()











