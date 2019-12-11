from lib import utils
import matplotlib
from matplotlib import pyplot as plt 
import numpy as np
from scipy.interpolate import spline
import scipy.signal as signal
import scipy.io as scio
# matplotlib.use('PDF')

def plot_Q_S_rewards():
    '''
    Q,SARSA, on graph1, alpha=0.01, 0,05, 0,15, 0,25
    '''
    graph = 2
    alphas = [0.01, 0.05, 0.15, 0.25]
    epoch = 3000
    save = True
    if save:
        matplotlib.use('PDF')

    Q_rewards_files = []
    S_rewards_files = []

    for alpha in alphas:
        Q_rewards_files.append(r'results\rewards\alpha{}_epsilon0.1_sim100_epoch{}_algoQ_graph{}_rewards.csv'.format(alpha, epoch, graph) )
        S_rewards_files.append(r'results\rewards\alpha{}_epsilon0.1_sim100_epoch{}_algoSARSA_graph{}_rewards.csv'.format(alpha, epoch, graph) )



    # Q_rewards_f = r'D:\MyCodes\RLSSP\results\alpha0.01_epsilon0.1_sim100_epoch1500_algoQ_graph2_rewards.csv'
    # S_rewards_f= r'D:\MyCodes\RLSSP\results\alpha0.01_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_rewards.csv'
    Q_rewards = []
    S_rewards = []
    for f in Q_rewards_files:
        reward = utils.load_csv_data(f, 'rewards').values
        Q_rewards.append( signal.medfilt(reward, 21) )
    for f in S_rewards_files:
        reward = utils.load_csv_data(f, 'rewards').values
        S_rewards.append( signal.medfilt(reward, 21) )


    

    # x = np.arange(0, len(Q_rewards[0]))
    # power_smooth = spline(x, Q_rewards[0], x)
    # midfilt = signal.medfilt(Q_rewards[1], 21)
    # plt.plot(Q_rewards[0], linewidth=0.5, color='blue')
    # plt.plot(power_smooth, linewidth=0.5, color='red')
    # plt.plot(midfilt, linewidth=0.5, color='red')
    
    # plt.scatter(x, Q_rewards, label='Q-ssp alpha=0.25', alpha=0.5)
    # plt.scatter(x, S_rewards, label='SARSA-ssp alpha=0.25', alpha=0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    c1 = '#023474'
    c2 = '#ef0107'

    # ax.plot(Q_rewards[0], c=c1, mec='none', ms=3, label='Q$_{ravg}$-SSPR $\\alpha_0=0.01$')
    # ax.plot(Q_rewards[1], c=c2, mec='none', ms=3, label='Q$_{ravg}$-SSPR $\\alpha_0=0.05$')
    # ax.plot(Q_rewards[2],  mec='none', ms=3, label='Q$_{ravg}$-SSPR $\\alpha_0=0.15$')
    # ax.plot(Q_rewards[3],  mec='none', ms=3, label='Q$_{ravg}$-SSPR $\\alpha_0=0.25$')

    # ax.plot(S_rewards[0],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha_0=0.01$')
    # ax.plot(S_rewards[1],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha_0=0.05$')
    # ax.plot(S_rewards[2],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha_0=0.15$')
    # ax.plot(S_rewards[3],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha_0=0.25$')

    ax.plot(Q_rewards[0], c=c1, mec='none', ms=3, label='Q$_{SSP}$, $\\alpha_0=0.01$')
    ax.plot(Q_rewards[1], c=c2, mec='none', ms=3, label='Q$_{SSP}$, $\\alpha_0=0.05$')
    ax.plot(Q_rewards[2],  mec='none', ms=3, label='Q$_{SSP}$, $\\alpha_0=0.15$')
    ax.plot(Q_rewards[3],  mec='none', ms=3, label='Q$_{SSP}$, $\\alpha_0=0.25$')

    ax.plot(S_rewards[0],  mec='none', ms=3, label='SARSA$_{SSP}$, $\\alpha_0=0.01$')
    ax.plot(S_rewards[1],  mec='none', ms=3, label='SARSA$_{SSP}$, $\\alpha_0=0.05$')
    ax.plot(S_rewards[2],  mec='none', ms=3, label='SARSA$_{SSP}$, $\\alpha_0=0.15$')
    ax.plot(S_rewards[3],  mec='none', ms=3, label='SARSA$_{SSP}$, $\\alpha_0=0.25$')

    ax.set_xlabel('number of learning episodes', fontsize=14)
    ax.set_ylabel('rewards', fontsize=14)

    ax.legend()
    # linewidth = 0.8
    # plt.plot(Q_rewards[0], linewidth=linewidth, label='Q-ssp alpha=0.01', alpha=1)
    # plt.plot(Q_rewards[1], linewidth=linewidth, label='Q-ssp alpha=0.05', alpha=1)
    # plt.plot(Q_rewards[2], linewidth=linewidth, label='Q-ssp alpha=0.15', alpha=1)
    # plt.plot(Q_rewards[3], linewidth=linewidth, label='Q-ssp alpha=0.25', alpha=1)

    # plt.plot(S_rewards[0], linewidth=linewidth, label='SARSA-ssp alpha=0.01', alpha=1)
    # plt.plot(S_rewards[1], linewidth=linewidth, label='SARSA-ssp alpha=0.05', alpha=1)
    # plt.plot(S_rewards[2], linewidth=linewidth, label='SARSA-ssp alpha=0.15', alpha=1)
    # plt.plot(S_rewards[3], linewidth=linewidth, label='SARSA-ssp alpha=0.25', alpha=1)

    # plt.xlabel('episodes')
    # plt.ylabel('rewards')

    # plt.legend()
    plt.show()
    if save:
        plt.savefig(r'figures\alpha-Q-SARSA-rewards-epoch{}-graph{}.pdf'.format(epoch, graph))

def plot_accuracy():
    '''
    
    on graph2, with own best params
    Q and SARSA, alpha=0.05; alg1,alg2,alg3,alg4, a=0.001.
    Q and SARSA, alpha=0.25; alg1,alg2,alg3,alg4, a=0.005.
    '''
    figure = 2
    save = True
    if save:
        matplotlib.use('PDF')

    LA_file1 = r'results/op-a0.001Iter30000.mat'
    LA_file2 = r'results/op-a0.005Iter30000.mat'
    file_Q_g2_005 = r'results/accuracy_padding/alpha0.05_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv'
    file_S_g2_005 = r'results/accuracy_padding/alpha0.05_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv'
    file_Q_g2_025 = r'results/accuracy_padding/alpha0.25_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv'
    file_S_g2_025 = r'results/accuracy_padding/alpha0.25_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv'

    LA_data1 = scio.loadmat(LA_file1)
    LA_data5 = scio.loadmat(LA_file2)

    d1_0001 = LA_data1['opt1'][0]
    d2_0001 = LA_data1['opt2'][0]
    d3_0001 = LA_data1['opt3'][0]
    d4_0001 = LA_data1['opt4'][0]
    d1_0005 = LA_data5['opt1'][0]
    d2_0005 = LA_data5['opt2'][0]
    d3_0005 = LA_data5['opt3'][0]
    d4_0005 = LA_data5['opt4'][0]

    Q_g2_005 = utils.load_csv_data(file_Q_g2_005, 'accuracy').values
    S_g2_005 = utils.load_csv_data(file_S_g2_005, 'accuracy').values
    Q_g2_025 = utils.load_csv_data(file_Q_g2_025, 'accuracy').values
    S_g2_025 = utils.load_csv_data(file_S_g2_025, 'accuracy').values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
  
    if figure == 1:
        # ax.plot(d1_0001, mec='none', ms=3, label='Algo1 $\\lambda=0.001$')
        # ax.plot(d2_0001, mec='none', ms=3, label='Algo2 $\\lambda=0.001$')
        # ax.plot(d3_0001, mec='none', ms=3, label='Algo3 $\\lambda=0.001$')
        # ax.plot(d4_0001, mec='none', ms=3, label='Algo4 $\\lambda=0.001$')
        #
        Q_g2_005 = signal.medfilt(Q_g2_005, 21)
        S_g2_005 = signal.medfilt(S_g2_005, 21)
        
        ax.plot(S_g2_005, color='red', mec='none', ms=3, label='SARSA$_{SSP}$ $\\alpha=0.05$')
        ax.plot(Q_g2_005, color='blue', mec='none', ms=3, label='Q$_{SSP}$ $\\alpha=0.05$')
        
    else:
        # ax.plot(d1_0005, mec='none', ms=3, label='Algo1 $\\lambda=0.005$')
        # ax.plot(d2_0005, mec='none', ms=3, label='Algo2 $\\lambda=0.005$')
        # ax.plot(d3_0005, mec='none', ms=3, label='Algo3 $\\lambda=0.005$')
        # ax.plot(d4_0005, mec='none', ms=3, label='Algo4 $\\lambda=0.005$')

        Q_g2_025 = signal.medfilt(Q_g2_025, 21)
        S_g2_025 = signal.medfilt(S_g2_025, 21)

        ax.plot(S_g2_025, color='red', mec='none', ms=3, label='SARSA$_{SSP}$ $\\alpha=0.25$')
        ax.plot(Q_g2_025, color='blue',  mec='none', ms=3, label='Q$_{SSP}$ $\\alpha=0.25$')


    ax.plot()
    ax.set_xlabel('number of learning episodes', fontsize=12)
    ax.set_ylabel('accuracy', fontsize=12)
    ax.legend()
    my_x_ticks = np.arange(0, 30000, 2000)
    plt.xticks(my_x_ticks)

    plt.show()

    # ax.plot(S_rewards[0],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha=0.01$')
    # ax.plot(S_rewards[1],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha=0.05$')
    # ax.plot(S_rewards[2],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha=0.15$')
    # ax.plot(S_rewards[3],  mec='none', ms=3, label='SARSA$_{ravg}$-SSPR $\\alpha=0.25$')
    if save:
        plt.savefig(r'figures\alpha-Q-SARSA-accuracy-graph2-fig{}.pdf'.format(figure))

def data_padding():
    '''
    padding 0s, for plotting
    files need padding:
    Only Run Once
    '''
    files = [
        r'results/accuracy/alpha0.05_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv',
        r'results/accuracy/alpha0.05_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv',
        r'results/accuracy/alpha0.25_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv',
        r'results/accuracy/alpha0.25_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv'
    ]
    files_padding = [
        r'results/accuracy_padding/alpha0.05_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv',
        r'results/accuracy_padding/alpha0.05_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv',
        r'results/accuracy_padding/alpha0.25_epsilon0.1_sim100_epoch1500_algoQ_graph2_accuracy.csv',
        r'results/accuracy_padding/alpha0.25_epsilon0.1_sim100_epoch1500_algoSARSA_graph2_accuracy.csv'
    ]
    
    for i in range(len(files)):
        file = files[i]; file_padding = files_padding[i]
        padding(file, file_padding)

def padding(file, file_padding):
    '''
    padding accuracy files
    '''
    data = plotting.load_csv_data(file, 'accuracy').values
    new_data = np.zeros((30000,))
    new_data[:len(data)] = data
    start = 0

    for i in reversed(range(len(data))):
        if data[i] != 0:
            start = i
            break
    
    np.random.seed()
    for i in range(start+1, 30000):
        new_data[i] = np.random.uniform(low=data[start], high=1.0)
    utils.save_csv_data(file_padding, {'accuracy':new_data})
    return True
    # return data

def plot_data_lists(data_list, 
                    label_list, 
                    length=10, 
                    height=6, 
                    x_label='x', 
                    y_label='y', 
                    label_fsize=14, 
                    save=True, 
                    figure_name='temp'):
    '''
    data_list: 应该为data1， data2， data3...等
    把这些data画在一张图上
    '''
    import matplotlib.pyplot as plt
    if save:
        matplotlib.use('PDF')

    fig, ax = plt.subplots(figsize=(length, height))
    ax.grid(True)

    for data, label in zip(data_list, label_list):
        ax.plot(data, label=label) # mec='none', ms=3, label='Algo1 $\\lambda=0.005$'
    
    ax.plot()
    ax.set_xlabel(x_label, fontsize=label_fsize)
    ax.set_ylabel(y_label, fontsize=label_fsize)
    ax.legend()
    ax.grid(True)
    
    if save:
        plt.savefig(figure_name)
    else:
        plt.show()

def plot_regrets():
    save = True
    graph = 1
    if save:
        matplotlib.use('PDF')
    
    result_dir = r'./results/regret_reserved/'

    if graph == 1:
        Q_regret_f = result_dir + r'regret_rlssp_Q_graph1_2019_05_25_14_46_53.csv'
        SARSA_regret_f = result_dir + r'regret_rlssp_SARSA_graph1_2019_05_25_15_54_10.csv'
        klhhr_regret_f = result_dir + r'regret_klhhr_graph1_2019_05_25_15_27_32.csv'
        cucb_regret_f = result_dir + r'regret_cucb_graph1_2019_05_25_15_27_35.csv'
        ts_regret_f = result_dir + r'regret_ts_graph1_2019_05_25_15_04_17.csv'
    elif graph == 2:
        Q_regret_f = result_dir + r'regret_rlssp_Q_graph2_2019_05_22_14_47_03.csv'
        SARSA_regret_f = result_dir + r'regret_rlssp_SARSA_graph2_2019_05_25_15_57_54.csv'
        klhhr_regret_f = result_dir + r'regret_klhhr_graph2_2019_05_22_15_59_58.csv'
        cucb_regret_f = result_dir + r'regret_cucb_graph2_2019_05_22_16_02_55.csv'
        ts_regret_f = result_dir + r'regret_ts_graph2_2019_05_25_14_16_37.csv'

    

    files = [Q_regret_f, SARSA_regret_f, klhhr_regret_f, cucb_regret_f, ts_regret_f]
    labels = ['$Q_{SSP}$, $\\alpha_0=0.25$', '$SARSA_{SSP}$, $\\alpha_0=0.25$', 'KL-HHR', 'CUCB', 'Thompson Sampling']
    # load数据
    datas = []
    x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]) # x：需要画到图上的数据点的位置
    for f in files:
        data = utils.load_csv_data(f, 'regret').values
        filtered = np.zeros(len(x)) # 需要画到图上的数据点
        for i, index in enumerate(x):
            filtered[i] = data[index//10] # //10 因为每隔10次记录一次
        datas.append(filtered)
    
    # 初始化图表
    fig, ax = plt.subplots(figsize=(10, 6)) # 
    fontsize = 14
    # ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    # 画图
    markers = ['o', 'p', 'v', 's', 'D']
    for i, filtered in enumerate(datas):
        ax.plot(x, filtered, marker=markers[i], mec='none', ms=6, label=labels[i])

    # 显示
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('number of received packets', fontsize=fontsize)
    ax.set_ylabel('regret', fontsize=fontsize)
    plt.show()

    if save:
        plt.savefig(r'figures\regret-graph{}.pdf'.format(graph))

    

if __name__ == "__main__":
    
    # plot_Q_S_rewards()
    plot_accuracy()
    # data_padding()
    # plot_regrets()
    
