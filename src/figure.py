import tensorflow as tf
import numpy as np
import quant_utils as qu
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque
import random

def get_weights_biases_histogram(sess, weight_name, q, bias_name=None):

    w = sess.graph.get_tensor_by_name(weight_name)
    w = sess.run(w)
    
    # Test quantize correctly    
    # unique, counts = np.unique(w, return_counts=True)

    if bias_name:
        b = sess.graph.get_tensor_by_name(bias_name)
        b = sess.run(b, feed_dict={})
        b = tf.constant(b, dtype=tf.float32)
        
    else :
        b = None

    if q:
        w_arr, s_arr = qu.quantize(w)
        w = tf.constant(w_arr, dtype=tf.float32)
    else:
        w = tf.constant(w, dtype=tf.float32)
        s = 0.

    weight = w_arr * s_arr
    unique, counts = np.unique(weight, return_counts=True)
    print("")
    # x = range(len(unique))
    # plt.bar(unique, counts, width=0.8, color="blue")
    # plt.xlim(xmin=-4.5, xmax = 4.5)
    # plt.savefig("fig.png")

    return w, b, s

def train_unique(sess):

    WEIGHT_NAME = 'layer1_1/dequant_conv_w1:0'
    LEVEL_NAME = 'layer1_1/Add_1:0'
    SCALE_NAME = 'layer1_1/truediv_2:0'
    w = sess.graph.get_tensor_by_name(WEIGHT_NAME)
    w = sess.run(w)

    l = sess.graph.get_tensor_by_name(LEVEL_NAME)
    l = sess.run(l)
    s = sess.graph.get_tensor_by_name(SCALE_NAME)
    s = sess.run(s)

    unique_l, counts_l = np.unique(l, return_counts=True)
    print("1) level unique: ", unique_l)
    print("--------------------------------------")
    print("1) level counts: ", counts_l)
    print("")

    unique_s, _ = np.unique(s, return_counts=True)
    print("2) scale: ", unique_s)
    print("")

    weight = l*s
    unique_w, _ = np.unique(weight, return_counts=True)
    print("3) mul weight: ", unique_w)
    print("")

    unique, counts = np.unique(w, return_counts=True)
    print("4) weight unique: ", unique)
    print("--------------------------------------")
    print("4) weight counts: ", counts)
    print("")

def model_unique():

    SAVE_DIR = 'Model/CIFAR10/alexnet_2/alexnet.ckpt.meta'
    CHEKPOINT = './Model/CIFAR10/alexnet_2/'
    WEIGHT_NAME = 'layer1/add:0'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(SAVE_DIR)
    saver.restore(sess, tf.train.latest_checkpoint(CHEKPOINT))

    tf.get_default_graph()

    w = sess.graph.get_tensor_by_name(WEIGHT_NAME)
    w = sess.run(w)

    unique, counts = np.unique(w, return_counts=True)
    print("unique: ", unique)
    print("======================================")
    print("counts: ", counts)
    print("")
    
def train_scale_with_range():

    SAVE_DIR = 'Model/CIFAR10/alexnet_2/alexnet.ckpt.meta'
    CHEKPOINT = './Model/CIFAR10/alexnet_2/'
    SCALE_NAME = 'layer1_1/truediv:0'
    RANGE_NAME = 'layer1_1/Max:0'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(SAVE_DIR)
    saver.restore(sess, tf.train.latest_checkpoint(CHEKPOINT))

    # tf.get_default_graph()

    s = sess.graph.get_tensor_by_name(SCALE_NAME)
    s = sess.run(s)

    r = sess.graph.get_tensor_by_name(RANGE_NAME)
    r = sess.run(r)

    unique_r, _ = np.unique(r, return_counts=True)
    print("======================================")
    print("1. range")
    print("unique: ", unique_r)
    print("--------------------------------------")

    unique_s, _ = np.unique(s, return_counts=True)
    print("2. scale")
    print("unique: ", unique_s)
    print("======================================")
    print("")
    # print("counts: ", counts)
    # print("")

def model_scale_with_range():

    SAVE_DIR = 'Model/CIFAR10/alexnet_2/alexnet.ckpt.meta'
    CHEKPOINT = './Model/CIFAR10/alexnet_2/'
    SCALE_NAME = 'layer1/truediv:0'
    RANGE_NAME = 'layer1/Max:0'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(SAVE_DIR)
    saver.restore(sess, tf.train.latest_checkpoint(CHEKPOINT))

    # tf.get_default_graph()

    s = sess.graph.get_tensor_by_name(SCALE_NAME)
    s = sess.run(s)

    r = sess.graph.get_tensor_by_name(RANGE_NAME)
    r = sess.run(r)

    unique_r, _ = np.unique(r, return_counts=True)
    print("======================================")
    print("1. range")
    print("unique: ", unique_r)
    print("--------------------------------------")

    unique_s, _ = np.unique(s, return_counts=True)
    print("2. scale")
    print("unique: ", unique_s)
    print("======================================")
    print("")
    # print("counts: ", counts)
    # print("")

def plot_scale(step, w1, w2, w3, w4, w5, w6, w7, w8, name):
    x = step

    plt.plot(x, w1, label='Conv1')
    plt.plot(x, w2, label='Conv2')
    plt.plot(x, w3, label='Conv3')
    plt.plot(x, w4, label='Conv4')
    plt.plot(x, w5, label='Conv5')
    plt.plot(x, w6, label='FC6')
    plt.plot(x, w7, label='FC7')
    plt.plot(x, w8, label='FC8')

    plt.xlabel('Epoch')
    plt.ylabel('# of level index')

    plt.legend()
    plt.savefig(name)

def plot_acc(step, model1, model2, name):
    import matplotlib.style
    import matplotlib as mpl
    mpl.style.use('classic')
    x = step

    plt.plot(x, model1, 'blue', label='2bit weights')
    if model2 is not None:
        plt.plot(x, model2, 'red', label='3bit weights')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')

    plt.legend(loc='lower right')
    plt.savefig(name, type="png")

def element_count_scale(sess, fix_model=None, tr_model=None):
    '''
        Should call it with plot_scale function!
    '''
    if tr_model is not None:
        w1 = sess.run(tr_model.extend_conv_w1_scale)
        w2 = sess.run(tr_model.extend_conv_w2_scale)
        w3 = sess.run(tr_model.extend_conv_w3_scale)
        w4 = sess.run(tr_model.extend_conv_w4_scale)
        w5 = sess.run(tr_model.extend_conv_w5_scale)
        w6 = sess.run(tr_model.extend_fc_w6_scale)
        w7 = sess.run(tr_model.extend_fc_w7_scale)
        w8 = sess.run(tr_model.extend_fc_w8_scale)

        s1 = sess.run(tr_model.extend_conv_w1)
        s2 = sess.run(tr_model.extend_conv_w2)
        s3 = sess.run(tr_model.extend_conv_w3)
        s4 = sess.run(tr_model.extend_conv_w4)
        s5 = sess.run(tr_model.extend_conv_w5)
        s6 = sess.run(tr_model.extend_fc_w6)
        s7 = sess.run(tr_model.extend_fc_w7)
        s8 = sess.run(tr_model.extend_fc_w8)

        unique_1, count_1 = np.unique(s1, return_counts=True)
        unique_2, _ = np.unique(s2, return_counts=True)
        unique_3, _ = np.unique(s3, return_counts=True)
        unique_4, _ = np.unique(s4, return_counts=True)
        unique_5, _ = np.unique(s5, return_counts=True)
        unique_6, _ = np.unique(s6, return_counts=True)
        unique_7, _ = np.unique(s7, return_counts=True)
        unique_8, _ = np.unique(s8, return_counts=True)

        return unique_1.size, unique_2.size, unique_3.size, unique_4.size, unique_5.size, unique_6.size, unique_7.size, unique_8.size

    if fix_model is not None:
        q1 = sess.run(fix_model.q_conv_w1)
        q2 = sess.run(fix_model.q_conv_w2)
        q3 = sess.run(fix_model.q_conv_w3)
        q4 = sess.run(fix_model.q_conv_w4)
        q5 = sess.run(fix_model.q_conv_w5)
        q6 = sess.run(fix_model.q_fc_w6)
        q7 = sess.run(fix_model.q_fc_w7)
        q8 = sess.run(fix_model.q_fc_w8)

        unique_1, count_1 = np.unique(q1, return_counts=True)
        unique_2, count_2 = np.unique(q2, return_counts=True)
        unique_3, count_3 = np.unique(q3, return_counts=True)
        unique_4, count_4 = np.unique(q4, return_counts=True)
        unique_5, count_5 = np.unique(q5, return_counts=True)
        unique_6, count_6 = np.unique(q6, return_counts=True)
        unique_7, count_7 = np.unique(q7, return_counts=True)
        unique_8, count_8 = np.unique(q8, return_counts=True)

        print("===========================================")
        print("index_1: ", unique_1)
        print("counts_1: ", count_1)
        print("-------------------------------------------")
        # print("scale: ", w1)
        print("index_2: ", unique_2)
        print("counts_2: ", count_2)
        print("-------------------------------------------")
        print("index_3: ", unique_3)
        print("counts_3: ", count_3)
        print("-------------------------------------------")
        print("index_4: ", unique_4)
        print("counts_4: ", count_4)
        print("-------------------------------------------")
        print("index_5: ", unique_5)
        print("counts_5: ", count_5)
        print("-------------------------------------------")
        print("index_6: ", unique_6)
        print("counts_6: ", count_6)
        print("-------------------------------------------")
        print("index_7: ", unique_7)
        print("counts_7: ", count_7)
        print("-------------------------------------------")
        print("index_8: ", unique_8)
        print("counts_8: ", count_8)
        print("-------------------------------------------")
        print("======================================")
        print("")

        return unique_1.size, unique_2.size, unique_3.size, unique_4.size, unique_5.size, unique_6.size, unique_7.size, unique_8.size

xlabel_fontdict = {'size':15}
ylabel_fontdict = {'size':14}
            
def plot_max_level():
    import matplotlib.style
    import matplotlib as mpl
    mpl.style.use('classic')

    epoch = []
    origin_1_2 = []
    for i in range(1,101):
        epoch.append(i)

    for i in range(1,101):
        origin_1_2.append(2)
    ori1_2 = np.array(origin_1_2)
    level1_2 = "/workspace/temp/1_2.npy"
    # origin_1_2 = "/workspace/temp/origin_1_2.npy"
    level2_3 = "/workspace/temp/2_3.npy"
    origin_2_3 = "/workspace/temp/origin_2_3.npy"
    level3_4 = "/workspace/temp/3_4.npy"
    origin_3_4 = "/workspace/temp/origin_3_4.npy"
    level4_5 = "/workspace/temp/4_5.npy"
    origin_4_5 = "/workspace/temp/origin_4_5.npy"
    num1_2 = np.load(level1_2)
    plt.plot(epoch, num1_2, 'c',label='1,2-bit dual', linewidth=2.5)
    # ori1_2 = np.load(origin_1_2)
    # plt.plot(epoch, ori1_2, 'c--',label='1/2-bit dual(2-bit)', linewidth=2.5)
    num2_3 = np.load(level2_3)
    plt.plot(epoch, num2_3, 'm', label='2,3-bit dual', linewidth=2.5)
    # ori2_3 = np.load(origin_2_3)
    # plt.plot(epoch, ori2_3, 'm--', label='2/3-bit dual(3-bit)', linewidth=2.5)
    num3_4 = np.load(level3_4)
    plt.plot(epoch, num3_4, 'g',label='3,4-bit dual', linewidth=2.5)
    # ori3_4 = np.load(origin_3_4)
    # plt.plot(epoch, ori3_4, 'g--', label='3/4-bit dual(4-bit)', linewidth=2.5)
    num4_5 = np.load(level4_5)
    plt.plot(epoch, num4_5, 'y', label='4,5-bit dual', linewidth=2.5)
    # ori4_5 = np.load(origin_4_5)
    # plt.plot(epoch, ori4_5, 'y--', label='4/5-bit dual(5-bit)', linewidth=2.5)

    plt.xlabel('Epoch')
    plt.ylabel('Maximum number of index levels', ylabel_fontdict)
    plt.yticks(np.arange(1, 32, step=3))

    plt.legend(loc='upper left')
    plt.savefig('level.png', type="png")

plot_max_level()
