import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

from variants import *
from lib import utils
from lib.utils import *

def make_epsilon_greedy_policy(Q, epsilon, paths):
    def policy_fn(state):
        '''
        根据Q[state]下各个可选的action，以及对应概率，以epsilon-greedy的方式选一个action
        不能选择在paths里面的action，因此需要对不在paths里面的action进行epsilon-greedy的选：先挑出action，再分配概率
        有可能无action可选，返回None。
        '''

        actions = [a for a in Q[state] if a not in paths] # 剔除已经在paths中的action（next node）
        nA = len(actions)
        action_probs = np.ones(nA, dtype=float) * epsilon/nA
        action_values = [Q[state][action] for action in actions]

        if 0 == len(actions):# 如果无action可选，比如一个节点只有一个出节点，而那个节点已经在paths里面了。
            return None
        else:
            best_action =np.argmax(action_values)
            action_probs[best_action] += 1 - epsilon
            epsilon_action = np.random.choice(a=np.array(actions), p=np.array(action_probs)) # 选action
            return epsilon_action

    return policy_fn    

def make_greedy_policy(Q, paths):
    def policy_g(state):
        '''
        选择当前state下有最大Q值的action
        不能选择在paths里面的action，因此需要先挑出不在path里面的action，然后再选argmax
        有可能无action可选，返回None。
        '''
        actions = [a for a in Q[state] if a not in paths]
        action_values = [Q[state][a] for a in actions]

        if 0 == len(actions):
            return None
        else:
            action_idx = np.argmax(action_values)
            best_action =  actions[action_idx]
            return best_action
    return policy_g




        
def display_Q(Q):
    for key in Q:
        print(key, Q[key])

class Env(object):
    def __init__(self, sgraph):
        self.sgraph = sgraph
        # self.current_state = self.sgraph.vs
        self.reward_history = initialize_Q(sgraph) # 记录state下这个action的平均reward
        self.action_num_history = initialize_Q(sgraph) # 记录state下这个action选择了多少次
        # self.max_length = np.max([self.sgraph.edges[state][a][0] for state in self.sgraph.edges for a in self.sgraph.edges[state] ]) # 所有可能的最长边，用来计算reward。
        self.max_length = sgraph.max_length

        self.temp = []

    
    def step(self, state, action):
        '''
        return next_state and reward
        '''
        next_state = action
        reward, raw_reward = self.compute_reward(state, action)
        return next_state, reward, raw_reward

    def compute_reward(self, state, action):
        lengths = self.sgraph.edges[state][action][0]
        lengths_probs = self.sgraph.edges[state][action][1]
        # print(state, action, np.sum(lengths_probs))
        sampled_length = np.random.choice(a=np.array(lengths), p=np.array(lengths_probs)) # 返回一个随机长度

        # 计算reward
        # instant_reward = self.max_length - sampled_length # sampled_length越长，reward越小。最长的sampled_length的reward是1，是0应该也可以
        instant_reward =  - sampled_length
        # 这里是历史平均reward，作为真正的reward来返回。
        actual_reward = (self.reward_history[state][action] * self.action_num_history[state][action] + instant_reward)/(self.action_num_history[state][action] + 1)

        # 
        if 9 == state:
            self.temp.append(actual_reward)


        # 更新reward_history和action_num_history
        self.reward_history[state][action] = actual_reward
        self.action_num_history[state][action] = self.action_num_history[state][action] + 1
        return actual_reward, instant_reward  ## 刚刚写成了sample_length!!

    def reset(self):
        # 这里reset不能把reward_history和action_num_history也清零了！！不然一个episode就几步，然后reward_history就被清零了
        # self.reward_history = initialize_Q(self.sgraph) # 初始化为0
        # self.action_num_history = initialize_Q(self.sgraph) # 初始化为0
        # 返回初始节点（作为state）
        return self.sgraph.vs


def initialize_Q(sgraph):
    '''
    通过一个随机图sgraph，初始化一个Q表，
    这个初始化过程，和初始化环境的reward_history, action_num_history的过程一样，
    所以也可以用来初始化环境的reward_history, action_num_history。

    Q的结构：
        Q: {s1:{a1:v, a2:v},
            s2:{a1:v, a3:v, a4:v},
            ...
            s_vd:{}
        }
    '''
    Q = {}
    edges = sgraph.edges
    vd = sgraph.vd
    for key in edges: # 一个node就是一个state
        start = key
        for dest in edges[key]:
            # start, dest = edge[0][0], edge[0][1]
            # lenths, prob = node[1], node[2]
            # anum = len(lenths)
            if start in Q: # 如果这个节点在Q中，就增加边
                Q[start][dest] = 0
            else: # 如果这个节点不在Q中，就创建新dict
                Q[start] = {dest: 0}
    if vd not in Q:
        Q[vd] = {vd:0} # 把vd节点添加一个指向自己的
    return Q

class Agent(object):
    def __init__(self, params, sgraph):
        self.sgraph = sgraph
        self.paths = [] # 选action的时候，选的action不能是paths中的节点，因为要避免环路。有可能无action可选，返回'no_action'。
        self.final_path = []
        self.Q = initialize_Q(self.sgraph)
        self.params = params # 'epsilon', 'alpha'
        self.behavior_policy = make_epsilon_greedy_policy(self.Q, self.params['epsilon'], self.paths) # 这个是函数，可以调用。调用后返回行为选择概率。
        self.learned_policy = make_greedy_policy(self.Q, self.final_path) # 这个也是函数，调用后直接返回最优行为。
        

    def update_Q(self, state, action, reward, next_state):
        alpha = self.params['alpha']
        gamma = self.params['gamma'] # gamma应该为1

        # next_state_values = [self.Q[next_state][action] for action in self.Q[next_state] if action not in self.paths ] # state下有最大Q值的action的Q值。这是greedy-policy得到的Q值。
        next_state_values = [self.Q[next_state][action] for action in self.Q[next_state] ]
        if 0 != len(next_state_values):
            maxQ = max(next_state_values)
        else:
            maxQ = 0
        
        td_target = reward + gamma*maxQ  # Q-learning的target
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] +  alpha*td_delta # Q-learning更新策略。

    def update_SARSA(self, state, action, reward, next_state, next_action):
        alpha = self.params['alpha']
        gamma = self.params['gamma'] # gamma应该为1

        td_target = reward + gamma*self.Q[next_state][next_action] # SARSA的td target
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] = self.Q[state][action] + alpha*td_delta


    def get_final_path(self, env):
        '''
        依据当前的Q table，greedy地得到path
        '''
        current_state = env.reset()
        terminal_state = self.sgraph.vd
        timestep = 0 # 防止死循环

        self.final_path.clear()
        self.final_path.append(current_state)
        path_reward = 0
        while current_state != terminal_state and timestep < 1000:
            greedy_action = self.learned_policy(current_state)

            if greedy_action == None:
                break
            next_state, reward, raw_reward = env.step(current_state, greedy_action)
            path_reward += raw_reward
            current_state = next_state
            timestep += 1
            self.final_path.append(current_state)

        return self.final_path, path_reward

    def reset(self):
        self.paths.clear()
        # self.paths = []
    



def run_rlssp_algorithm(params):
    '''
    训练过程
    '''

    # 初始化一个随机图实例
    if 1 == params['graph']:
        sto_graph = StoGraph(graph1_vs, graph1_vd, graph1_vnum, graph1_edges, graph1_right_path)
    else:
        sto_graph = StoGraph(graph2_vs, graph2_vd, graph2_vnum, graph2_edges, graph2_right_path)
    # 初始化一个环境
    env = Env(sto_graph)
    # 初始化一个agent
    agent = Agent(params, sto_graph)

    get_right_path = False
    episode_states = utils.EpisodeStats(
        episode_lengths = np.zeros(params['episode_num']),
        episode_rewards = np.zeros(params['episode_num']),
        episode_raw_rewards = np.zeros(params['episode_num']),
        recent_reward_mean = np.zeros(params['episode_num']),
        convergence_num = np.zeros((1,)),
        AVI = np.zeros((1,)),
        TS = np.zeros((1,)),
        SPS = np.zeros((1,))
    )

    # 交互
    convergence_counter = 0 # 若recent_reward_mean连续10次超过收敛阈值，则认为收敛，这个convergence_counter就是用来计数的，超过十就认为收敛，记录下i_episode。
    convergence = False

    now = datetime.now() # current date and time
    timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    total_regret = 0
    regret_record = [0,]
    regret_record_file = r'./results/regret/regret_rlssp_{}_graph{}_{}.csv'.format(params['algorithm'], params['graph'], timestr)
    optimal_reward = (-1) * agent.sgraph.shortest_expected_length

    for i_episode in range(params['episode_num']):
        # 初始化
        current_state = env.reset()
        agent.reset()
        terminal_state = sto_graph.vd
        timestep = 0 # 一个episode最长timestep限制，防止死循环

        # 一个episode
        agent.paths.append(current_state)

        if params['algorithm'] == 'Q':
            while current_state != terminal_state and timestep <= 1000:
                # 选出action
                # actions = [a for a in agent.Q[current_state]] # 当前state下可以选择的action
                # actions必须是不在路径中的action？挑选出可选的action，然后再赋予概率，进行挑选
                epsilon_action = agent.behavior_policy(current_state)

                # 交互
                if epsilon_action == None: # 当前节点无action可选
                    break 

                next_state, reward, raw_reward = env.step(current_state, epsilon_action)

                # agent更新Q
                agent.update_Q(current_state, epsilon_action, reward, next_state)
                
                # 转到下一个状态
                current_state = next_state

                # 记录信息
                episode_states.episode_rewards[i_episode] += reward
                episode_states.episode_raw_rewards[i_episode] += raw_reward
                episode_states.episode_lengths[i_episode] += 1
                
                
                agent.paths.append(current_state)
                timestep += 1
                
        
        elif params['algorithm'] == 'SARSA':
            action = agent.behavior_policy(current_state)

            while current_state != terminal_state and timestep <= 1000:
                
                next_state, reward, raw_reward = env.step(current_state, action)

                next_action = agent.behavior_policy(next_state)
                if next_action == None: # 到达不了vd了，这个episode也结束
                    break
                agent.update_SARSA(current_state, action, reward, next_state, next_action)

                current_state = next_state
                action = next_action

                episode_states.episode_rewards[i_episode] += reward
                episode_states.episode_lengths[i_episode] += 1
                
                agent.paths.append(current_state)
                timestep += 1
        # 一个episode结束 #

        # 累加这一个episode的SPS个数（这一个episode有几条边来自right path的采样）
        for edge in zip(agent.paths[:-1], agent.paths[1:]):
            if edge in sto_graph.right_edges:
                episode_states.SPS[0] += 1

        # 计算近10个episode的平均reward
        episode_states.recent_reward_mean[i_episode] = np.mean(episode_states.episode_rewards[max(0, i_episode-9): i_episode])

        # 这个episode后是否收敛
        if not convergence:
            if params['graph'] == 1:
                if episode_states.recent_reward_mean[i_episode] >= -18:
                    convergence_counter += 1
                else:
                    convergence_counter = 0 # 否则重新计数
            elif params['graph'] == 2:
                if episode_states.recent_reward_mean[i_episode] >= -68:
                    convergence_counter += 1
                else:
                    convergence_counter = 0
            # 连续大于
            if convergence_counter >= 10:
                convergence = True
                episode_states.convergence_num[0] = i_episode - 10
        
        # 收敛则不进行下一个episode了
        if convergence and params['convergence_break']: # 收敛即停止运行，由convergence_break控制
            break

        # 检查是否到达了vd，因为有可能是到了一个死结点而退出while的
        # if current_state == sto_graph.vd:
        #     episode_states.episode_success[0] += 1

        # regret
        path, path_reward = agent.get_final_path(env)
        print(path)
        
        optimal_reward = (-1)*agent.sgraph.shortest_expected_length
        
        regret = optimal_reward - path_reward
        total_regret += regret
        print('Episode {}/{}, optimal Rw:{}, actual Rw:{}, regret:{}, total regret:{}'.format(
            i_episode+1, params['episode_num'], optimal_reward, path_reward, regret, total_regret))
        
        if (i_episode+1) % 10 ==0:
            regret_record.append(total_regret)
        if (i_episode+1) % 1000 ==0:
            save_csv_data(regret_record_file, {'regret': regret_record})

    # 所有episode结束，或者已经收敛
    episode_states.AVI[0] = episode_states.convergence_num[0] 
    episode_states.TS[0] = np.sum(episode_states.episode_lengths) # 所有TS-total samples(timestep)数


    agent.get_final_path(env) # 存下final path
    if agent.final_path == env.sgraph.right_path:
        get_right_path = True

    if params['display']: # print info
        print('max_length:', sto_graph.max_length)
        display_Q(agent.Q)
        
        print('Final path:', agent.final_path)
        print('convergence episodes:', episode_states.convergence_num[0])
        print('Finished!')

        plt.plot(episode_states.episode_rewards, color='green', linewidth=1, label='rewards')
        plt.plot(episode_states.recent_reward_mean, color='red', linewidth=1, label='rewards_mean')
        plt.axvline(x=episode_states.convergence_num[0], color='blue', linewidth=1)
        plt.legend()
        plt.show()

    
    return episode_states, get_right_path, agent.final_path


def run_rlssp_algorithm_accuracy(params):
    '''
    在训练完成后，得到agent的final path。重复repeat_num次，以得到固定episode_num下的准确率。
    '''
    params['display'] = False
    episode_num_max = params['episode_num_max']
    accuracy = np.zeros((episode_num_max,))

    # save
    filename = 'alpha{}_epsilon{}_sim{}_epoch{}_algo{}_graph{}_accuracy.csv'.format(params['alpha'], params['epsilon'], params['repeat_num'], params['episode_num_max'], params['algorithm'], params['graph'])
    filename = os.path.join(params['results_path'], filename)

    for episode_num in range(episode_num_max): # 每一个episode_num下的得到正确path的ratio, 为accuracy.
        params['episode_num'] = episode_num
        # params['episode_num'] = 80

        right_num = 0
        repeat_num = params['repeat_num']

        for i_simulation in range(repeat_num):
            global random_seed
            np.random.seed(random_seed); random_seed += 1
            episode_states, get_right_path, _ = run_rlssp_algorithm(params)
            if get_right_path: # 这一次simulation得到了正确的path
                right_num += 1
        accuracy[episode_num] = right_num/repeat_num

        print('. Episode num:{}, accuracy:{}'.format(episode_num, accuracy[episode_num]) )
        if episode_num % 10 == 0:
            utils.save_csv_data(filename, {'accuracy':accuracy}) # save

        if np.sum(accuracy[max(0, episode_num-50): episode_num]) >= 49:
            break
    
    # save
    utils.save_csv_data(filename, {'accuracy':accuracy})
    # plot
    # plt.plot(accuracy)
    # plt.show()
    print('Finished!')

def run_rlssp_algorithm_rewards(params):
    '''
    训练episode_num_max次，得到每一个episode后的reward值。重复repeat_num次进行平均。
    '''
    episode_num_max = params['episode_num_max']

    params['display'] = False
    params['episode_num'] = episode_num_max # 每一次simulation，都进行episode_num_max个episode进行学习
    
    rewards = np.zeros((episode_num_max,))

    repeat_num = params['repeat_num']
    for i_simulation in range(repeat_num):
        global random_seed
        np.random.seed(random_seed); random_seed += 1
        print('\rSimulation {}/{}.'.format(i_simulation+1, repeat_num) )
        episode_states, _, _ = run_rlssp_algorithm(params)
        rewards  = rewards + episode_states.episode_rewards # 这里是np的array的+，是element-wise的+
        
    rewards /= params['repeat_num'] # element-wise的/

    # save rewards
    filename = 'alpha{}_epsilon{}_sim{}_epoch{}_algo{}_graph{}_rewards.csv'.format(params['alpha'], params['epsilon'], params['repeat_num'], params['episode_num_max'], params['algorithm'], params['graph'])
    filename = os.path.join(params['results_path'], filename)
    utils.save_csv_data(filename, {'rewards':rewards})

    # plot
    # plt.plot(rewards)
    # plt.show()
    print('Finished!')

def run_rlssp_algorithm_regret(params):
    MAXIMUM = 150000
    params['episode_num'] = MAXIMUM
    params['display'] = False
    episode_states, _, final_path = run_rlssp_algorithm(params)



def run_rlssp_algorithm_AVI_PS_TS_SPS(params):
    '''
    训练直到收敛！得到每一次收敛后的AVI, TS, SPS值。重复repeat_num次进行平均。
    收敛后就停止运行。

    AVI: average number of inerations until convergence
    PS: 正确convergence time/总实验次数。100次收敛实验，几次正确收敛了
    TS: total number of samples(timesteps) from a graph, until convergence。相当于总timesteps个数
    SPS: number of samples(timesteps) from the optimal path, until convergence。多少timestep采到了right path上的边
    '''
    params['convergence_break'] = True
    params['display'] = False
    params['episode_num'] = 5000 # 2000次肯定大于收敛需要的次数，防止在收敛前就达到episode_num次了

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    AVI = np.zeros((len(alphas),)) 
    PC = np.zeros((len(alphas),)) 
    TS = np.zeros((len(alphas),)) 
    SPS = np.zeros((len(alphas),)) 
    SPS_TS = np.zeros((len(alphas),)) 

    repeat_num = params['repeat_num']
    for i, alpha in enumerate(alphas):
        params['alpha'] = alpha
        print('alpha={}'.format(alpha))
        avi = 0; ts = 0; sps = 0; pc = 0
        for i_simulation in range(repeat_num):
            global random_seed
            np.random.seed(random_seed); random_seed += 100
            
            episode_states, get_right_path, _ = run_rlssp_algorithm(params)
            if get_right_path:
                pc += 1
            
            avi = episode_states.AVI[0]; AVI[i] += avi
            ts = episode_states.TS[0]; TS[i] += ts
            sps = episode_states.SPS[0]; SPS[i] += sps

            print('\rSimulation {}/{}. AVI={} TS={} SPS={} PC={}'.format(i_simulation+1, repeat_num, avi, ts, sps, pc) )
            # plt.plot(episode_states.episode_rewards)
            # plt.show()
            pass

        # 取100次simulation的平均
        AVI[i] /= repeat_num
        TS[i] /= repeat_num
        SPS[i] /= repeat_num
        PC[i] = pc
        SPS_TS[i] = SPS[i]/TS[i]

        # save 
        filename = r'AVI_PC_TS_SPS_SPSTS_algo{}_graph{}.csv'.format(params['algorithm'], params['graph'])
        filename = os.path.join(params['results_path'], filename)
        data_dict = {'alpha':alphas,
                    'AVI':AVI,
                    'PC':PC,
                    'TS':TS,
                    'SPS':SPS,
                    'SPS_TS':SPS_TS}
        utils.save_csv_data(filename, data_dict)
    

    print('Finished')

if __name__ == "__main__":
    params = {
    'gamma':1,
    'alpha':0.25,
    'epsilon':0.1,
    'episode_num':2000, # 单独运行run_rlssp_algorithm时需要的
    'episode_num_max':1500,
    'repeat_num':100, # 测试episode_num=i时，其正确率，需要运行多少次episode_num=i的实验
    'results_path':r'./results',
    'display':True,
    'algorithm':'SARSA', # Q 或者 'SARSA'
    'graph':2, # 随机图1或2
    'convergence_break': False # 是否收敛后就停止
    }
    global random_seed
    np.random.seed(random_seed); random_seed += 1
    # ###### 测试rewards ########
    # for graph in [1, 2]:
    #     for algo in ['Q', 'SARSA']:
    #         for alpha in [0.01, 0.05, 0.15, 0.25, 0.35]:
    #             print('graph {}, algo={}, alpha={}'.format(graph, algo, alpha))
    #             params['graph'] = graph; params['algorithm'] = algo; params['alpha'] = alpha
    #             run_rlssp_algorithm_rewards(params)
    
    # ######## 测试accuracy ########
    # for graph in [2]:
    #     for algo in ['Q', 'SARSA']:
    #         for alpha in [0.25, 0.05]:
    #             print('graph {}, algo={}, alpha={}'.format(graph, algo, alpha))
    #             params['graph'] = graph; params['algorithm'] = algo; params['alpha'] = alpha
    #             run_rlssp_algorithm_accuracy(params)

    ####### 测试AVI, PS, TS, SPS ########
    # run_rlssp_algorithm_AVI_PS_TS_SPS(params)

    # graph = 2
    # print('graph {}, algo={}, alpha={}'.format(graph, 'Q', 0.05))
    # params['graph'] = graph; params['algorithm'] = 'Q'; params['alpha'] = 0.05
    # run_rlssp_algorithm_accuracy(params)

    # params['convergence_break'] = True
    # run_rlssp_algorithm(params)
    run_rlssp_algorithm_regret(params)