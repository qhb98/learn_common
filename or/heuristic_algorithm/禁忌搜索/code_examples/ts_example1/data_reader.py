# coding: utf-8
# @FileName: :data_reader.py
# @Time: 2022/8/6 15:27
# @Author: QHB
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import re


class Data:
    """
    生成Solomon标准算例
    """

    def __init__(self):
        # customer 的数量
        self.customer_num = None
        # depot 的数量
        self.node_num = None
        self.vehicle_num = None
        self.capacity = None
        self.cor_X = []
        self.cor_Y = []
        self.demand = []
        self.ready_time = []
        self.due_time = []
        self.service_time = []
        self.dis_matrix = {}

    def read_data(self, path, customer_num, depot_num):
        """
        从.txt文件中读取solomon标准算例
        Args:
            path:
            customer_num:
            depot_num:

        Returns:

        """
        self.customer_num = customer_num
        self.node_num = customer_num + depot_num
        f = open(path, "r")
        lines = f.readlines()
        count = 0
        for line in lines:
            count += 1
            if count == 5:
                line = line[:-1].strip()
                re_str = re.split(r"+", line)
                self.vehicle_num = int(re_str[0])
                self.capacity = float(re_str[1])
            elif 10 <= count <= 10 + customer_num:
                line = line[:-1]
                re_str = re.split(r"+", line)
                self.cor_X.append(float(re_str[2]))
                self.cor_Y.append(float(re_str[3]))
                self.demand.append(float(re_str[4]))
                self.ready_time.append(float(re_str[5]))
                self.due_time.append(float(re_str[6]))
                self.service_time.append(float(re_str[7]))

        self.dis_matrix = {}
        for i in range(0, self.node_num):
            dis_temp = {}
            for j in range(0, self.node_num):
                dis_temp[j] = int(math.hypot(self.cor_X[j], self.cor_Y[i] - self.cor_Y[j]))
            self.dis_matrix[i] = dis_temp

    def plot_nodes(self):
        """
        绘图方法
        Returns:

        """
        graph = nx.DiGraph()
        nodes_name = [str(x) for x in list(range(self.node_num))]
        graph.add_nodes_from(nodes_name)
        cor_xy = np.array([self.cor_X, self.cor_Y]).T.astype(int)
        pos_location = {nodes_name[i]: x for i, x in enumerate(cor_xy)}
        nodes_color_dict = ["r"] + ["gray"] * (self.node_num - 1)
        nx.draw_networkx(graph, pos_location, node_size=200, nodes_color=nodes_color_dict, labels=None)
        plt.show(graph)

    def plt_route(self, route, color="k"):
        graph = nx.DiGraph()
        nodes_name = [0]
        cor_xy = [[self.cor_X[0], self.cor_Y[0]]]
        edge = []
        edges = [[0, route[0]]]
        for i in route:
            nodes_name.append(i)
            cor_xy.append([self.cor_X[i], self.cor_Y[i]])
            edge.append(i)
            if len(edge) == 2:
                edges.append(copy.deepcopy(edge))
                edge.pop(0)
        edges.append([route[-1], 0])

        graph.add_nodes_from(nodes_name)
        graph.add_edges_from(edges)
        pos_location = {nodes_name[i]: x for i, x in enumerate(cor_xy)}
        nodes_color_dict = ["r"] + ["gray"] * (len(route))
        nx.draw_networkx(graph, pos_location, nodes_size=200, nodes_color=nodes_color_dict, edge_color=color,
                         labels=None)
        plt.show(graph)
