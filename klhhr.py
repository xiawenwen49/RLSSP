'''
KL Hop-by-Hop Routing for Stochastic Shortest Path Finding
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

class HHR(object):
    '''
    Hop-by-Hop routing
    '''
    def __init__(self, sgraph):
        self.theta = initialize_Q(sgraph) # 记录每条边的θ估计值，
        self.N = 0 # 模拟所有边被用的总次数
        self.packet = 0 # 发送的packet数/寻路（一整个path）次数
        self.regret = 0 # 累积regret
        self.w = initialize_Q(sgraph) # 记录每条边的index
        self.J = np.zeros(sgraph.vnum + 1) # J(v):节点v到目标节点的munimum cumulative index，1是第一个节点，所以长度要+1
        self.sgraph = sgraph
        self.average_delay = initialize_Q(sgraph) # 记录每一条边的历史length的均值

        # 这个次数，是成功才算一次。
        # 因为比如说这次length是18，那么这个18应该看作重传次数，t要加18.
        # 而这里num_history只+1，这个num_history只是为了计算平均average_delay,从而估计θ
        self.num_history = initialize_Q(sgraph) 
        self.t = initialize_Q(sgraph) # 这里把length看作重传次数，每一次加的是length（次）

    def find_path(self):
        '''
        return: path, reward, regret
        '''
        current_node = self.sgraph.vs 
        vd = self.sgraph.vd 
        path = list()
        path.append(current_node)
        reward = 0 # 这次寻找path，一共得到的reward
        while current_node != vd:
            
            adjacent_nodes = list(self.sgraph.edges[current_node].keys()) # 当前节点的相邻节点
            next_node_indexs = np.zeros(len(adjacent_nodes)) # index，是相当于length
            # 计算相邻节点的index值
            for i, node in enumerate(adjacent_nodes):
                next_node_indexs[i] = self.w[current_node][node] + self.J[node]
            # 选择下一个节点
            argmin_ = np.argmin(next_node_indexs)
            next_node = adjacent_nodes[argmin_]
            path.append(next_node)

            
            # next_node!!! 写成node了！！！
            sampled_length = np.random.choice(a=np.array(self.sgraph.edges[current_node][next_node][0]), p=np.array(self.sgraph.edges[current_node][next_node][1]))
            # sampled_length看作是delay，或重试的次数
            self.N += sampled_length # 加的是sampled_length
            self.t[current_node][next_node] += sampled_length # 这个边加sampled_length（次）
            
            self.updata_theta(current_node, next_node,sampled_length)
            self.updata_w(current_node, next_node)
            self.updata_J()
            
            reward += (-1) * sampled_length # 最后是负的
            current_node = next_node
        return path, reward

    def run(self, params):
        MAXIMUM = 150000
        now = datetime.now() # current date and time
        timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
        # np.random.seed(10000) # 就设置一次？？
        # optimal_reward = -40 # 很重要，这个等于最优路径能得到的reward。等于最短期望路径长度的负
        regret_record = [0,]
        regret_record_file = r'./results/regret/regret_klhhr_graph{}_{}.csv'.format(params['graph'], timestr)
        optimal_reward = (-1) * self.sgraph.shortest_expected_length
        for packet_i in range(MAXIMUM):
            path, reward = self.find_path()
            # total_reward += reward
            regret = optimal_reward - reward
            self.regret += regret
            
            print(path)
            print('packet:{}, optimal Rw:{}, actual Rw:{}, regret:{}, total regret:{}'.format(packet_i+1, optimal_reward, reward, regret, self.regret))

            # record regret & reward?
            if (packet_i+1) % 10 == 0:
                regret_record.append(self.regret)
            if (packet_i+1) % 1000 == 0:
                save_csv_data(regret_record_file, {'regret': regret_record})

    def updata_theta(self, current_node, next_node,sampled_length):
        '''
        update the estimated successful probability of an edge/link: (current_node, next_node)
        '''
        # 历史平均值
        average_delay = (self.average_delay[current_node][next_node] * self.num_history[current_node][next_node] + sampled_length) / (self.num_history[current_node][next_node] + 1)
        self.average_delay[current_node][next_node] = average_delay
        self.num_history[current_node][next_node] += 1
        # θ = 1/历史平均长度（尝试次数）
        self.theta[current_node][next_node] = 1 / average_delay # delay的倒数，作为是成功概率。也就是说认为delay/length就是当作尝试次数
        
    def updata_w(self, current_node, next_node):
        '''
        updata index of an edge/link: (current_node, next_node)
        '''
        # self.w[current_node][next_node] = sp.optimize.line_search()
        f2 = np.log(self.N) + 4 * np.log(np.log(self.N) + 1e-6)
        f2 = f2 / (self.t[current_node][next_node] ) # 公式中的f2/t_i，KL进行line search的最大的限制值
        max_u = linesearch(self.theta[current_node][next_node], constraint=f2) # self.theta[current_node][next_node]是当前估计出的θ
        self.w[current_node][next_node] = 1 / max_u # 倒数
        
    def updata_J(self):
        '''
        update the minimun cumulative index of every node to destination
        把self.w当作边长，寻找最短路径 的长度
        '''
        # w更新过后，weighted_edges也需要更新
        self.weighted_edges = generate_weighted_edges(self.sgraph, self.w)
        for source in self.sgraph.vertices:
            distance, _ = bellman_ford(self.sgraph.vertices, self.weighted_edges, source)
            self.J[source] = distance[self.sgraph.vd] # source节点到目标节点的最小距离。index为距离
                  
def generate_weighted_edges(sgraph, w):
    '''
    sgraph：一个stochastic graph对象
    w：目前为止计算的的每一条边的index
    return：[[from, to, weight],
            [from, to, weight],
            ...
            ]
    '''
    weighted_edges = []
    for from_ in list(sgraph.edges.keys()):
        for to_ in list(sgraph.edges[from_].keys()):
            weighted_edges.append( (from_, to_, w[from_][to_]) )
    return weighted_edges

def main():
    params = {
        'graph':1
    }

    if params['graph'] == 1:
        sto_graph = StoGraph(graph1_vs, graph1_vd, graph1_vnum, graph1_edges, graph1_right_path)
    else:
        sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    
    hhr = HHR(sto_graph)
    hhr.run(params)

if __name__ == "__main__":
    # test_1()
    # test_2()
    # test_3()
    main()
