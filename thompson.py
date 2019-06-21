'''
Thompson Sampling (TS) for Stochastic Shortest Path finding
    input: stochastic graph
    output: path, reward, regret

'''
import numpy as np 
import scipy as sp
import scipy.optimize
from datetime import datetime
import argparse
import logging

from variants import *
from lib.utils import *
from rlssp import initialize_Q
from klhhr import generate_weighted_edges

parser = argparse.ArgumentParser(description='Thompson Sampling (TS) for Stochastic Shortest Path finding')
parser.add_argument('--num-epoch', dest='epoch', type=int, default=150000, help='number of training epochs')
parser.add_argument('--graph', dest='graph', type=int, default=2, choices=[1, 2], help='which graph to run on')
parser.add_argument('--algo', dest='algo', type=str, default='ts', choices=['Q', 'SARSA', 'klhhr', 'cucb', 'ts'], help='use which algorithm')

class TS(object):
    '''
    choose path using Thompson Sampling algorithm
    '''
    def __init__(self, sgraph):
        self.sgraph = sgraph

        self.paths = sgraph.all_paths # 所有path，一个path是一个arm
        # self.alpha = np.ones(len(sgraph.all_paths), dtype=np.int32) # 维护的θ符合的后验Beta概率分布的参数α。 初始化不能=0；α=β=1时，Beta分布相当于[0, 1]上的均匀分布
        # self.beta = np.ones(len(sgraph.all_paths), dtype=np.int32) # 维护的θ符合的后验Beta概率分布的参数β
        self.miu = initialize_Q(sgraph) 
        self.sigma = initialize_Q(sgraph) # θ本身符合的分布的方差
        self.sigma_ = None # length符合的分布的方差
        # self.mean_reward = 0 # 当前为止所有path得到的reward的均值，也就是平均length*(-1)
        # self.num_reward = 0 # 当前为止得到的reward的次数，用来计算平均reward的
        self.regret = 0
        self.reward = 0
        self.initialize()

    def initialize(self):
        '''
        初始化
        miu，sigma，sigma_都设为1
        '''
        for v1 in self.miu.keys():
            for v2 in self.miu[v1].keys():
                self.miu[v1][v2] = 1

        for v1 in self.sigma.keys():
            for v2 in self.sigma[v1].keys():
                self.sigma[v1][v2] = 1
        
        self.sigma_ = 1

    def run(self, params):
        MAXIMUM = 150000
        now = datetime.now() # current date and time
        timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
        # np.random.seed(100001) # 就设置一次？？
        # optimal_reward = -40 # 很重要，这个等于最优路径能得到的reward。等于最短期望路径长度的负
        regret_record = [0,]
        reward_record = [0,]
        regret_record_file = r'./results/regret/regret_ts_graph{}_{}.csv'.format(params['graph'], timestr)
        optimal_reward = (-1) * self.sgraph.shortest_expected_length

        for packet_i in range(MAXIMUM):
            path, reward = self.find_path()

            regret = optimal_reward - reward
            self.regret += regret
            self.reward += reward
            
            print(path)
            print('ts, packet:{}, optimal Rw:{}, actual Rw:{}, regret:{}, total regret:{}'.format(packet_i+1, optimal_reward, reward, regret, self.regret))

            # record regret & reward?
            if (packet_i+1) % 10 == 0:
                regret_record.append(self.regret)
                reward_record.append(self.reward)
            if (packet_i+1) % 1000 == 0:
                save_csv_data(regret_record_file, {'regret': regret_record})

    def find_path(self):
        '''
        choose a path for this packet
        '''
        # 采样出每一个edge的值
        w = initialize_Q(self.sgraph)
        for v1 in w.keys():
            for v2 in w[v1].keys():
                ln_theta_e = np.random.normal(self.miu[v1][v2], self.sigma[v1][v2])
                theta_e = np.exp(ln_theta_e)
                w[v1][v2] = theta_e
        
        # 依据采出的每条边的θ值，最短路径找path
        weighted_edges = generate_weighted_edges(self.sgraph, w)
        
        distance, predecessor = bellman_ford(self.sgraph.vertices, weighted_edges, self.sgraph.vs)
        path = [self.sgraph.vd,] # 从目标节点往源节点找
        while path[-1] != self.sgraph.vs:
            pre_node = predecessor[path[-1]]
            path.append(pre_node)
        
        path = path[::-1]

        # 走path，得到实际长度、reward
        reward = 0
        y_te = np.zeros(len(path)) # y_te: [L_(v1,v2), L_(v2,v3), ... L_(vn, vd)]，所以是path的长度少一个
        for i, (v1, v2) in enumerate(zip(path[:-1], path[1:]) ): # 这个长度相当于是现实采出来的，依据自己预设的分布
            sampled_length = np.random.choice(a=np.array(self.sgraph.edges[v1][v2][0]), p=np.array(self.sgraph.edges[v1][v2][1]))
            reward += (-1) * sampled_length
            y_te[i] = sampled_length
        
        # 更新path上的边的μe，σe
        for i, (v1, v2) in enumerate(zip(path[:-1], path[1:]) ):
            old_miu = self.miu[v1][v2]
            old_sigma = self.sigma[v1][v2]
            y_te_ = y_te[i]
            sigma_ = self.sigma_

            new_miu = ( old_miu/(old_sigma**2) + (np.log(y_te_) + sigma_**2/2)/(sigma_**2) ) / (1/old_sigma**2 + 1/sigma_**2) 
            new_sigma = np.sqrt( 1/(1/old_sigma**2 + 1/sigma_**2) )

            self.miu[v1][v2] = new_miu
            self.sigma[v1][v2] = new_sigma
        

        # 更新历史所有arm/path的平均length
        # self.mean_reward = (self.mean_reward * self.num_reward + reward) / (self.num_reward + 1)
        # self.num_reward += 1 

        return path, reward
        
def main(args):
    params = {
        'graph':args.graph
    }

    if params['graph'] == 1:
        sto_graph = StoGraph(graph1_vs, graph1_vd, graph1_vnum, graph1_edges, graph1_right_path)
    else:
        sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    
    ts = TS(sto_graph)
    ts.run(params)

if __name__ == "__main__":
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    
    args = parser.parse_args()
    logging.info(args)
    pass

    # main(args)



