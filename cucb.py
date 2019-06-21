'''
Combinatorial Upper Confidence Bound (CUCB) for Stochastic Shortest Path finding
    input: stochastic graph
    output: path, reward, regret
'''

import numpy as np 
import scipy as sp
import scipy.optimize
from datetime import datetime

from variants import *
from lib.utils import *
from rlssp import initialize_Q

class CUCB(object):
    '''
    choose path using CUCB algorithm
    '''
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.theta = initialize_Q(sgraph) # theta是估计的每条边的1/delay
        self.t = initialize_Q(sgraph) # 每条边被选择的次数
        self.paths = sgraph.all_paths
        self.path_index = np.zeros(len(self.paths))
        self.N = 0
        self.regret = 0

        self.average_delay = initialize_Q(sgraph) # 记录每一条边的历史length的均值
        self.num_history = initialize_Q(sgraph) # 记录每一条边的历史被选择次数

    def run(self, params):
        MAXIMUM = 150000
        now = datetime.now() # current date and time
        timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
        # np.random.seed(10000) # 就设置一次？？
        # optimal_reward = -40 # 很重要，这个等于最优路径能得到的reward。等于最短期望路径长度的负
        regret_record = [0,]
        regret_record_file = r'./results/regret/regret_cucb_graph{}_{}.csv'.format(params['graph'], timestr)
        optimal_reward = (-1) * self.sgraph.shortest_expected_length

        for packet_i in range(MAXIMUM):
            path, reward = self.find_path()
            # re-compute path index
            self.recompute_path_index()

            regret = optimal_reward - reward
            self.regret += regret

            print(path)
            print('cucb, packet:{}, optimal Rw:{}, actual Rw:{}, regret:{}, total regret:{}'.format(packet_i+1, optimal_reward, reward, regret, self.regret))

            # record regret & reward?
            if (packet_i+1) % 10 == 0:
                regret_record.append(self.regret)
            if (packet_i+1) % 1000 == 0:
                save_csv_data(regret_record_file, {'regret': regret_record})

    
    def find_path(self):
        argmin_ = np.argmin(self.path_index)
        path = self.paths[argmin_]
        reward = 0
        for v1, v2 in zip(path[:-1], path[1:]):
            sampled_length = np.random.choice(a=np.array(self.sgraph.edges[v1][v2][0]), p=np.array(self.sgraph.edges[v1][v2][1]))
            reward += (-1) * sampled_length

            # updata theta, t, N
            self.update_theta(v1, v2, sampled_length)
            self.t[v1][v2] += sampled_length # !!! sampled_length?
            self.N += sampled_length # !!! sampled_length?
        return path, reward

    
    def update_theta(self, v1, v2, sampled_length):
        # 历史平均值
        average_delay = (self.average_delay[v1][v2] * self.num_history[v1][v2] + sampled_length) / (self.num_history[v1][v2] + 1)
        self.average_delay[v1][v2] = average_delay
        self.num_history[v1][v2] += 1 # 这里加的是1，只是为了计算average_delay

        self.theta[v1][v2] = 1 / average_delay
    
    def recompute_path_index(self):
        for i, path in enumerate(self.paths): # 更新所有边的path
            self.path_index[i] = self.compute_path_index(path)
    
    def compute_path_index(self, path):
        index = 0
        for v1, v2 in zip(path[:-1], path[1:]): # 计算一条边的index
            theta = self.theta[v1][v2]
            n = self.N
            t_i = self.t[v1][v2]
            index += 1/(theta + np.sqrt(1.5 * np.log(n+1e-6) / (t_i+1e-6) ) + 1e-6 ) # 分母，log都不能出现0
        return index

def main():
    params = {
        'graph':1
    }

    if 1 == params['graph']:
        sto_graph = StoGraph(graph1_vs, graph1_vd, graph1_vnum, graph1_edges, graph1_right_path)
    else:
        sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    
    cucb = CUCB(sto_graph)
    cucb.run(params)


if __name__ == "__main__":
    # test_1()
    main()