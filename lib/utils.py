import matplotlib
import numpy as np 
import pandas as pd 
from collections import namedtuple


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_raw_rewards", "recent_reward_mean",
 "convergence_num", "AVI", "TS", "SPS"])

# class stochastic graph
class StoGraph(object):
    '''
    应该有节点个数，源节点，目标节点，每一个节点的出度，每一个节点的[出边目标节点（有向图）、长度、概率]，
    所有可能边长度的最大值，
    '''
    def __init__(self, vs, vd, vnum, edges, right_path):
        self.vs = vs
        self.vd = vd
        self.vnum = vnum # 节点个数
        self.enum = len(edges) # 边个数
        self.vertices = list(range(1, vnum+1)) # 具体的节点的编号，是一个list
        # self.right_path = right_path
        self.edges = initialize_dict(edges, vd) # 用edges字典存储所有节点其出边的长度与概率分布。
        # 计算出来的正确路径，与每条路径的期望长度，以及随机最短路径的长度
        self.right_path, self.expected_length, self.shortest_expected_length = compute_right_path(vs, vd, vnum, edges) 
        self.right_edges = [edge for edge in zip(self.right_path[:-1], self.right_path[1:])]
        self.all_paths = list() 

        self.get_all_paths(vs, vd, vnum, self.edges) # 保存所有从vs到vd的loop-free的path

        # initialize degree
        self.degree = {}
        for item in edges:  
            if item[0][0] in self.degree: # start node这个node在不在degree字典里面
                self.degree[item[0][0]] = self.degree[item[0][0]] + 1
            else: self.degree[item[0][0]] = 1

        self.max_length = np.max([self.edges[state][a][0] for state in self.edges for a in self.edges[state] ]) # 所有可能的最长边，用来计算reward。
    

    def get_all_paths(self, vs, vd, vnum, edges):
        '''
        edges：是dict
        dfs找到所有vs到vd的loop-free的path
        '''
        path = []
        visited = [False]*(self.vnum+1) # 节点编号从1开始

        self.dfs(vs, vd, edges, path, visited) 

    def dfs(self, vs, vd, edges, path, visited):
        '''
        vs:当前所在节点
        vd：目标节点
        path：前面已经有的路径
        visited：已经访问过的情况
        '''
        visited[vs] = True
        path.append(vs)
        if vs == vd:
            self.all_paths.append(list(path))  # 加list的原因，对path内容进行一个拷贝，不然python中的这个path是一个指针，后面对path的操作pop，会影响all_paths中的path。
        else:
            for i in edges[vs].keys():
                if visited[i] == False:
                    self.dfs(i, vd, edges, path, visited)
        path.pop()
        visited[vs] = False
        

def initialize_dict(raw_edges, vd):
    '''
    输入原始edges
    返回一个edges dict，值初始化为0
    
        返回的edges dict的结构
        edges:{
            node1:{adjacent_node1:[[lengths], [probs]], 
                    adjacent_node2:[[lengths], [probs]]
                    },
            node2:{adjacent_node1:[[lengths], [probs]]
                    },
            ...
        } 
    '''
    edges = {}
    for item in raw_edges: # edges是传进来的信息
        # initialize edges
        if item[0][0] in edges:
            edges[item[0][0]][item[0][1]] = [item[1], item[2]] # edges[start][dest] = [[length list], [probabillity list]]
        else:
            edges[item[0][0]] = {item[0][1]: [item[1], item[2]]}
    if vd not in edges: # 有可能vd节点没有出边，这里edges里面最好加上vd节点
        edges[vd] = {}

    return edges



def compute_right_path(vs, vd, vnum, edges):
    '''
    依据每条边的expected_length，计算最短路径
    edges:就是原始的edges list，反而更方便
    weighted_edges: bellman-ford的输入
    '''
    vertices = list(range(1, vnum+1)) 
    weighted_edges = list()
    expected_length = initialize_dict(edges, vd)
    for item in edges:
        from_to_, length, dist = item[0], item[1], item[2]
        length_ = np.sum(np.array(length)*np.array(dist)) # 期望长度
        weighted_edges.append( (from_to_[0], from_to_[1], length_) )
        expected_length[from_to_[0]][from_to_[1]] = length_

    # 寻找正确路径
    distance, predecessor = bellman_ford(vertices, weighted_edges, vs)
    path = [vd,]
    while path[-1] != vs:# 依据predecessor寻找最优路径
        pre_node = predecessor[path[-1]]
        path.append(pre_node)
    
    return path[::-1], expected_length, distance[vd] #反转路径



 
def save_csv_data(filename, data_dict):
    '''
    data_dict: {
        key: value, # 一列
        key: value  # 一列

    }
    '''
    df = pd.DataFrame(data_dict)
    df.to_csv(filename)

def load_csv_data(filename, key):
    df = pd.read_csv(filename)
    value = df[key]
    return value

def dijkstra():
    pass

def bellman_ford(vertices, edges, source):
    '''
    find the shortest distance for source to all other nodes.
    example:
    vertices: [1, 2, ...]
    edges: [(from, to, weight),
            (from, to, weight),
            ...]
    source: 1
    '''
    distance = np.zeros(len(vertices) + 1)
    predecessor = list(range(len(vertices) + 1))
    # initialize
    for vertex in vertices:
        distance[vertex] = np.inf
        predecessor[vertex] = None
    distance[source] = 0

    # relex edges repeatedly
    for i in range(len(vertices)-1):
        for edge in edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u
    
    # check negative-weight cycles
    return distance, predecessor

def KL(lambda_, u_):
    return lambda_ * np.log(lambda_/u_) + (1 - lambda_) * np.log((1-lambda_)/(1-u_))

def linesearch(lambda_, constraint, stepsize=0.0001):
    res = 0
    for u in np.arange(lambda_, 1, stepsize):
        if KL(lambda_, u) >= constraint:
            res = u
            break
    return res - stepsize