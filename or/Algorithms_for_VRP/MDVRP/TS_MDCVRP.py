# -*- coding: utf-8 -*-

import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import csv
import matplotlib.pyplot as plt


class Sol():
    def __init__(self):
        self.node_id_list = None
        self.obj = None
        self.action_id = None
        self.routes = None


class Node():
    def __init__(self):
        self.id = 0
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0
        self.depot_capacity = 15


class Model():
    def __init__(self):
        self.best_sol = None
        self.demand_dict = {}
        self.depot_dict = {}
        self.demand_id_list = []
        self.distance_matrix = {}
        self.tabu_list = None
        self.TL = 30
        self.opt_type = 0
        self.vehicle_cap = 0


def readCsvFile(demand_file, depot_file, model):
    with open(demand_file, 'r') as f:
        demand_reader = csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)

    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.depot_capacity = float(row['capacity'])
            model.depot_dict[node.id] = node


def calDistance(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
        for _, depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist


def generateInitialSol(node_id_list):
    node_id_list_ = copy.deepcopy(node_id_list)
    random.seed(0)
    random.shuffle(node_id_list_)
    return node_id_list_


def createActions(n):
    action_list = []
    nswap = n // 2
    # Swap
    for i in range(nswap):
        action_list.append([1, i, i + nswap])
    # DSwap
    for i in range(0, nswap, 2):
        action_list.append([2, i, i + nswap])
    # Reverse
    for i in range(0, n, 4):
        action_list.append([3, i, i + 3])
    return action_list


def doAction(node_id_list, action):
    node_id_list_ = copy.deepcopy(node_id_list)
    if action[0] == 1:
        index_1 = action[1]
        index_2 = action[2]
        node_id_list_[index_1], node_id_list_[index_2] = node_id_list_[index_2], node_id_list_[index_1]
        return node_id_list_
    elif action[0] == 2:
        index_1 = action[1]
        index_2 = action[2]
        temporary = [node_id_list_[index_1], node_id_list_[index_1 + 1]]
        node_id_list_[index_1] = node_id_list_[index_2]
        node_id_list_[index_1 + 1] = node_id_list_[index_2 + 1]
        node_id_list_[index_2] = temporary[0]
        node_id_list_[index_2 + 1] = temporary[1]
        return node_id_list_
    elif action[0] == 3:
        index_1 = action[1]
        index_2 = action[2]
        node_id_list_[index_1:index_2 + 1] = list(reversed(node_id_list_[index_1:index_2 + 1]))
        return node_id_list_


def selectDepot(route, depot_dict, model):
    min_in_out_distance = float('inf')
    index = None
    for _, depot in depot_dict.items():
        if depot.depot_capacity > 0:
            in_out_distance = model.distance_matrix[depot.id, route[0]] + model.distance_matrix[route[-1], depot.id]
            if in_out_distance < min_in_out_distance:
                index = depot.id
                min_in_out_distance = in_out_distance
    if index is None:
        print("there is no vehicle to dispatch")
    route.insert(0, index)
    route.append(index)
    depot_dict[index].depot_capacity = depot_dict[index].depot_capacity - 1
    return route, depot_dict


def splitRoutes(node_id_list, model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    depot_dict = copy.deepcopy(model.depot_dict)
    for node_id in node_id_list:
        if remained_cap - model.demand_dict[node_id].demand >= 0:
            route.append(node_id)
            remained_cap = remained_cap - model.demand_dict[node_id].demand
        else:
            route, depot_dict = selectDepot(route, depot_dict, model)
            vehicle_routes.append(route)
            route = [node_id]
            num_vehicle = num_vehicle + 1
            remained_cap = model.vehicle_cap - model.demand_dict[node_id].demand

    route, depot_dict = selectDepot(route, depot_dict, model)
    vehicle_routes.append(route)

    return num_vehicle, vehicle_routes


def calRouteDistance(route, model):
    distance = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        distance += model.distance_matrix[from_node, to_node]
    return distance


def calObj(node_id_list, model):
    num_vehicle, vehicle_routes = splitRoutes(node_id_list, model)
    if model.opt_type == 0:
        return num_vehicle, vehicle_routes
    else:
        distance = 0
        for route in vehicle_routes:
            distance += calRouteDistance(route, model)
        return distance, vehicle_routes


def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.show()


def outPut(model):
    work = xlsxwriter.Workbook('result.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, 'opt_type')
    worksheet.write(1, 0, 'obj')
    if model.opt_type == 0:
        worksheet.write(0, 1, 'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1, 1, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        worksheet.write(row + 2, 0, 'v' + str(row + 1))
        r = [str(i) for i in route]
        worksheet.write(row + 2, 1, '-'.join(r))
    work.close()


def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord = [model.depot_dict[route[0]].x_coord]
        y_coord = [model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0] == 'd1':
            plt.plot(x_coord, y_coord, marker='o', color='black', linewidth=0.5, markersize=5)
        elif route[0] == 'd2':
            plt.plot(x_coord, y_coord, marker='o', color='orange', linewidth=0.5, markersize=5)
        else:
            plt.plot(x_coord, y_coord, marker='o', color='b', linewidth=0.5, markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()


def run(demand_file, depot_file, epochs, v_cap, opt_type):
    model = Model()
    model.vehicle_cap = v_cap
    model.opt_type = opt_type
    readCsvFile(demand_file, depot_file, model)
    calDistance(model)
    action_list = createActions(len(model.demand_id_list))
    model.tabu_list = np.zeros(len(action_list))
    history_best_obj = []
    sol = Sol()
    sol.node_id_list = generateInitialSol(model.demand_id_list)
    sol.obj, sol.routes = calObj(sol.node_id_list, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    for ep in range(epochs):
        local_new_sol = Sol()
        local_new_sol.obj = float('inf')
        for i in range(len(action_list)):
            if model.tabu_list[i] == 0:
                new_sol = Sol()
                new_sol.node_id_list = doAction(sol.node_id_list, action_list[i])
                new_sol.obj, new_sol.routes = calObj(new_sol.node_id_list, model)
                new_sol.action_id = i
                if new_sol.obj < local_new_sol.obj:
                    local_new_sol = copy.deepcopy(new_sol)
        sol = local_new_sol
        for i in range(len(action_list)):
            if i == sol.action_id:
                model.tabu_list[sol.action_id] = model.TL
            else:
                model.tabu_list[i] = max(model.tabu_list[i] - 1, 0)
        if sol.obj < model.best_sol.obj:
            model.best_sol = copy.deepcopy(sol)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s" % (ep, epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)


if __name__ == '__main__':
    demand_file = r'../data/MDCVRP/demand.csv'
    depot_file = r'../data/MDCVRP/depot.csv'
    run(demand_file, depot_file, 300, 80, 1)
