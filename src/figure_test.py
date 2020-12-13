import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def plot_acc(index1, nob1, index2, nob2, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(index1, nob1, width=0.1, label='2bit Original Model')
    ax1.set_xticks([-2, -1, 0, 1])
    ax1.set_ylim([0, 7300])
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_ylim([0, 7300])
    ax2.bar(index2, nob2, width=0.2, label='3bit Upscaled Model')
    # x = index

    # plt.bar(x, nob, width=0.1, label='2bit Original Model')
    # # plt.plot(x, model2, label='Proposed Model')

    #ax1.set_ylabel('Number of observations')
    ax2.set_xlabel('Index levels', fontsize=16)
    # ax.set_ylabel('Number of observations')
    fig.text(0.03, 0.5, 'Number of observations', ha='center', va='center', rotation='vertical', fontsize=18)

    # plt.legend()
    plt.savefig(name)


origin_index = [-2, -1, 0, 1]
origin_nob = [838, 4418, 6950, 5218]

extend_indx = [-4, -3, -2, -1, 0, 1, 2, 3]
extend_nob = [496,  329, 2578, 1845, 4057, 2953, 2956, 2210]
save1_dir = './origin.png'
save2_dir = './extend.png'
plot_acc(origin_index, origin_nob, extend_indx, extend_nob, save1_dir)
# plot_acc(extend_indx, extend_nob, save2_dir)