import numpy as np 
import scipy as sp
import scipy.optimize
from datetime import datetime

from variants import *
from lib.utils import *
from rlssp import initialize_Q
import plot

def longestPalindrome(s: str) -> str:
    maxlen = 1
    start = 0
    for i in range(len(s)):
        if i - maxlen >= 1 and s[i-maxlen-1:i+1] == s[i-maxlen-1:i+1][::-1]:
            start = i - maxlen - 1
            maxlen += 2
            continue
        if i - maxlen >= 0 and s[i-maxlen:i+1] == s[i-maxlen:i+1][::-1]:
            start = i -maxlen
            maxlen += 1
    return s[start:start+maxlen] 

def convert(s: str, numRows: int) -> str:
    if numRows == 1:
        return str
    r = numRows
    group = len(s)//(r + r - 2)
    remainder = len(s)%(r + r - 2)
    if remainder != 0:
        group += 1
    groupstr = [[] for i in range(group)]
    for i in range(group-1):
        groupstr[i] = s[i*(r+r-2):(i+1)*(r+r-2)]
    groupstr[group-1] = s[(group-1)*(r+r-2):]
    
    result = []
    for i in range(r):
        for group in groupstr:
            if i == 0:
                index = 0
                if(index<len(group)):
                    result.append(group[index])
            elif i==r-1:
                index = r-1
                if(index<len(group)):
                    result.append(group[index])
            else:
                index1 = i
                index2 = r-1 + (r-1-i)
                if(index1<len(group)):
                    result.append(group[index1])
                if(index2<len(group)):
                    result.append(group[index2])
    
    restr = ""
    for s in result:
        restr += s
    return restr

def test_1():
    edges = [(1, 2, 1),
            (1, 3, 4),
            (2, 3, 5),
            (2, 4, 1),
            (2, 5, 7),
            (3, 2, 6),
            (3, 5, 8),
            (4, 5, 1),
            ]
    vertices = [1, 2, 3, 4, 5]
    source = 1
    distance, pre = bellman_ford(vertices, edges, source)
    print(distance)
    print(pre)

def test_2():
    graph = 2
    if graph == 1:
        sto_graph = StoGraph(graph1_vs, graph1_vd, graph1_vnum, graph1_edges, graph1_right_path)
    elif graph == 2:
        sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    
    print(sto_graph.right_path)
    print(len(sto_graph.all_paths))
    for i in sto_graph.all_paths:
        print(i)
    # print(sto_graph.all_paths)

def test_3():
    sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    total_reward = 0
    total_regret = 0
    optimal_reward = (-1)*sto_graph.shortest_expected_length

    # global random_seed
    # np.random.seed(random_seed); random_seed += 1
    for i in range(1, 150000):  
        reward = 0
        # global random_seed
        # np.random.seed(random_seed); random_seed += 1
        for edge in sto_graph.right_edges:
            v1, v2 = edge[0], edge[1]
            # global random_seed
            # np.random.seed(random_seed); random_seed += 1
            sampled_length = np.random.choice(a=np.array(sto_graph.edges[v1][v2][0]), p=np.array(sto_graph.edges[v1][v2][1]))
            reward += (-1)*sampled_length
        regret = optimal_reward - reward
        total_reward += reward
        total_regret += regret
        print('iter:{}, optimal Rw:{}, actual Rw:{}, regret:{}, mean reward:{}, total regret:{}'.format(i, optimal_reward, reward, regret, total_reward/i, total_regret) )
    print(sto_graph.right_edges)

def test_a():
    sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    print(sto_graph.enum)

def test_agent():
    agent = Agent(graph1_vs, graph1_vd, graph1_vnum, graph1_edges)
    for start in agent.Q:
        print(start, agent.Q[start])

def make_policy(Q):
    def policy(state):
        # return Q
        return state in Q
    return policy

def test_make_policy():
    Q = []
    p = make_policy(Q)
    print( p(1))
    Q.append(1)
    # print(Q)
    print(p(1))
    Q.clear()
    print( p(1) )

    # Q[2] = 790098 # 这里，只要是Q[i] = new_value这种更新方式，p就使用的还是一个Q。如果Q = []，这种方式p就用的不是一个Q
    # print(p(state))

def test_list():
    paths = [1, 3, 5]
    q = [1, 2, 3, 4, 6]

    actions = [a for a in q if a not in paths]
    print(actions)
    
def test_choice():
    lengths = [4.6, 6.4, 7.6, 8.9]
    lengths_probs = [0.4, 0.1, 0.2, 0.3]
    mean = 0
    num = 0
    for i in range(2000):
        sampled_length = np.random.choice(a=np.array(lengths), p=np.array(lengths_probs))
        # action = np.random.choice(a=[], p=[])
        # print(np.argmax(np.array([])))
        # print(action)
        mean = (mean*num + sampled_length)/(num+1)
        num += 1
        print(sampled_length, mean)
        
def test_comput_len():
    # 计算每条条边的期望长度
    new_edges = []
    for edge in graph1_edges:
        temp = [l*p for l, p in zip(edge[1], edge[2]) ]
        mean = np.sum(np.array(temp))
        new_edge = [edge[0], mean]
        new_edges.append(new_edge)
        print(new_edge)
    
def test_save_load():
    # save
    Q_rewards_f = r'D:\MyCodes\RLSSP\results\alpha0.25_epsilon0.1_sim100_epoch1500_algoSARSA_graph1_rewards.csv'
    # data_dict = {'nodes':graph2_right_path}
    # plotting.save_csv_data(filename, data_dict)
    # load
    # data = plotting.load_csv_data(filename, 'rewards')
    Q_rewards = utils.load_csv_data(Q_rewards_f, 'rewards')
    print(Q_rewards.values)

def test_load():
    filename = 'graph2_right_path.csv'
    filename = os.path.join(params['results_path'], filename)
    
    data = utils.load_csv_data(filename, 'nodes')
    values = data.values
    print(values)

def test_np():
    a = np.ones((5,))
    b = np.ones((5,))*7
    c = a + b
    print(c)
    c %= 2
    d = 7
    print(d%2)
def test_seed():
    seed = 9999
    for i in range(5):
        print(seed); seed += 1
def test_4():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]
    params = 0
    plot.plot_data_lists(params, list1, list2)

def test_5():
    for i in range(100000000000):
        if i < -1:
            pass

if __name__ == "__main__":
    # s = "PAYPALISHIRING"
    # numRows = 4
    # result = convert(s, numRows)
    # print(result)
    # test_3()
    # test_2()
    test_5()