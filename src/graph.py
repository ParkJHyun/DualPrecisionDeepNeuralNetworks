
import tensorflow as tf
import sys
import utils as util

class AlexNet_linear:
    def __init__(self, X_img, FLAGS, dropout, block, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.qnum = FLAGS.num_bit
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit
    
    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse ):
            with tf.variable_scope('layer1', reuse=self.reuse):
                self.conv_w1, self.conv_b1, B = util.conv(B, 64, self.qnum, bias=0.0, c_s=2, pool=True)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('conv_w1', self.conv_w1)
            self.parameter_list.append(self.conv_w1)
            self.parameter_list.append(self.conv_b1)

            with tf.variable_scope('layer2', reuse=self.reuse):
                self.conv_w2, self.conv_b2, B = util.conv(B, 192, self.qnum, bias=1.0, pool=True)
            self.parameter_list.append(self.conv_w2)
            self.parameter_list.append(self.conv_b2)

            with tf.variable_scope('layer3', reuse=self.reuse):
                self.conv_w3, self.conv_b3, B = util.conv(B, 384, self.qnum, bias=1.0, vg_drop=self.dropout)
            self.parameter_list.append(self.conv_w3)
            self.parameter_list.append(self.conv_b3)

            with tf.variable_scope('layer4', reuse=self.reuse):
                self.conv_w4, self.conv_b4, B = util.conv(B, 256, self.qnum, bias=1.0, vg_drop=self.dropout)
            self.parameter_list.append(self.conv_w4)
            self.parameter_list.append(self.conv_b4)

            with tf.variable_scope('layer5', reuse=self.reuse):
                self.conv_w5, self.conv_b5, B = util.conv(B, 256, self.qnum, bias=1.0, pool=True, pd='SAME')
            self.parameter_list.append(self.conv_w5)
            self.parameter_list.append(self.conv_b5)

            with tf.variable_scope('layer6', reuse=self.reuse):
                self.fc_w6, self.fc_b6, B = util.dense(B, 4096, self.qnum, self.dropout, fully=True)
            self.parameter_list.append(self.fc_w6)
            self.parameter_list.append(self.fc_b6)

            with tf.variable_scope('layer7', reuse=self.reuse):
                self.fc_w7, self.fc_b7, B = util.dense(B, 4096, self.qnum, self.dropout)
            self.parameter_list.append(self.fc_w7)
            self.parameter_list.append(self.fc_b7)

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w8, self.fc_b8, self.hypothesis = util.dense(B, self.num_class, self.qnum, self.dropout, hypothesis=True)
            self.parameter_list.append(self.fc_w8)
            self.parameter_list.append(self.fc_b8)
        return self.hypothesis

    def upgrade(self, X_img):
        B = X_img
        with tf.variable_scope('Train', reuse=self.reuse):
            with tf.variable_scope('Exlayer1', reuse=self.reuse):
                self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                    util.extend_bit_quant_conv(B, self.parameter_list, self.qnum, c_s=2, pool=True, expend_bit=self.up_bit, count=0)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w1', self.extend_conv_w1)

            # Convolution layer 2
            with tf.variable_scope('Exlayer2', reuse=self.reuse):
                self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                    util.extend_bit_quant_conv(B, self.parameter_list, self.qnum, pool=True, expend_bit=self.up_bit, count=2)
    
            # Convolution layer 3
            with tf.variable_scope('Exlayer3', reuse=self.reuse):
                self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                    util.extend_bit_quant_conv(B, self.parameter_list, self.qnum, expend_bit=self.up_bit, dropout=self.dropout, count=4)

            # Convolution layer 4
            with tf.variable_scope('Exlayer4', reuse=self.reuse):
                self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                    util.extend_bit_quant_conv(B, self.parameter_list, self.qnum, expend_bit=self.up_bit, dropout=self.dropout, count=6)

            # Convolution layer 5
            with tf.variable_scope('Exlayer5', reuse=self.reuse):
                self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                    util.extend_bit_quant_conv(B, self.parameter_list, self.qnum, pool=True, pd='SAME', expend_bit=self.up_bit, count=8)

            # Fully connected layer 1
            with tf.variable_scope('Exlayer6', reuse=self.reuse):
                self.extend_fc_w6, self.extend_fc_w6_scale, B = \
                    util.extend_bit_quant_dense(B, self.parameter_list, self.qnum, self.dropout, fully=True, expend_bit=self.up_bit, count=10)

            # Fully connected layer 2
            with tf.variable_scope('Exlayer7', reuse=self.reuse):
                self.extend_fc_w7, self.extend_fc_w7_scale, B = \
                    util.extend_bit_quant_dense(B, self.parameter_list, self.qnum, self.dropout, expend_bit=self.up_bit, count=12)

            # Fully connected layer 3
            with tf.variable_scope('Exlogits', reuse=self.reuse):
                self.extend_fc_w8, self.extend_fc_w8_scale, self.hypothesis = \
                    util.extend_bit_quant_dense(B, self.parameter_list, self.qnum, self.dropout, hypothesis=True, expend_bit=self.up_bit, count=14)
            
        return self.hypothesis

class AlexNet_log:
    def __init__(self, X_img, FLAGS, dropout, block, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.qnum = FLAGS.num_bit
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit
        self.lookup = tf.linspace(0.0, -7.0, 2**self.qnum)

    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('layer1', reuse=self.reuse):
                self.conv_w1, self.conv_b1, B = util.log_conv(B, 64, self.qnum, self.lookup, bias=0.0, c_s=2, pool=True)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('conv_w1', self.conv_w1)
            self.parameter_list.append(self.conv_w1)
            self.parameter_list.append(self.conv_b1)

            with tf.variable_scope('layer2', reuse=self.reuse):
                self.conv_w2, self.conv_b2, B = util.log_conv(B, 192, self.qnum, self.lookup, bias=1.0, pool=True)
            self.parameter_list.append(self.conv_w2)
            self.parameter_list.append(self.conv_b2)

            with tf.variable_scope('layer3', reuse=self.reuse):
                self.conv_w3, self.conv_b3, B = util.log_conv(B, 384, self.qnum, self.lookup, bias=1.0)
            self.parameter_list.append(self.conv_w3)
            self.parameter_list.append(self.conv_b3)

            with tf.variable_scope('layer4', reuse=self.reuse):
                self.conv_w4, self.conv_b4, B = util.log_conv(B, 256, self.qnum, self.lookup, bias=1.0)
            self.parameter_list.append(self.conv_w4)
            self.parameter_list.append(self.conv_b4)

            with tf.variable_scope('layer5', reuse=self.reuse):
                self.conv_w5, self.conv_b5, B = util.log_conv(B, 256, self.qnum, self.lookup, bias=1.0, pool=True, pd='SAME')
            self.parameter_list.append(self.conv_w5)
            self.parameter_list.append(self.conv_b5)

            with tf.variable_scope('layer6', reuse=self.reuse):
                self.fc_w6, self.fc_b6, B = util.log_dense(B, 4096, self.qnum, self.lookup, self.dropout, fully=True)
            self.parameter_list.append(self.fc_w6)
            self.parameter_list.append(self.fc_b6)

            with tf.variable_scope('layer7', reuse=self.reuse):
                self.fc_w7, self.fc_b7, B = util.log_dense(B, 4096, self.qnum, self.lookup, self.dropout)
            self.parameter_list.append(self.fc_w7)
            self.parameter_list.append(self.fc_b7)

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w8, self.fc_b8, self.hypothesis = util.log_dense(B, self.num_class, self.qnum, self.lookup, self.dropout, hypothesis=True)
            self.parameter_list.append(self.fc_w8)
            self.parameter_list.append(self.fc_b8)

        return self.hypothesis
    
    def upgrade(self, X_img):
        B = X_img
        with tf.variable_scope('Train', reuse=self.reuse):
            with tf.variable_scope('Exlayer1', reuse=self.reuse):
                self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                    util.extend_bit_log_quant_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                    c_s=2, pool=True, expend_bit=self.up_bit, count=0)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w1', self.extend_conv_w1)

            # Convolution layer 2
            with tf.variable_scope('Exlayer2', reuse=self.reuse):
                self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                    util.extend_bit_log_quant_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                    pool=True, expend_bit=self.up_bit, count=2)
    
            # Convolution layer 3
            with tf.variable_scope('Exlayer3', reuse=self.reuse):
                self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                    util.extend_bit_log_quant_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                    expend_bit=self.up_bit, count=4, dropout=self.dropout)

            # Convolution layer 4
            with tf.variable_scope('Exlayer4', reuse=self.reuse):
                self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                    util.extend_bit_log_quant_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                    expend_bit=self.up_bit, count=6, dropout=self.dropout)

            # Convolution layer 5
            with tf.variable_scope('Exlayer5', reuse=self.reuse):
                self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                    util.extend_bit_log_quant_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                    pool=True, pd='SAME', expend_bit=self.up_bit, count=8)

            # Fully connected layer 1
            with tf.variable_scope('Exlayer6', reuse=self.reuse):
                self.extend_fc_w6, self.extend_fc_w6_scale, B = \
                    util.extend_bit_log_quant_dense(B, self.parameter_list, self.qnum, self.lookup, self.dropout, 
                                    fully=True, expend_bit=self.up_bit, count=10)

            # Fully connected layer 2
            with tf.variable_scope('Exlayer7', reuse=self.reuse):
                self.extend_fc_w7, self.extend_fc_w7_scale, B = \
                    util.extend_bit_log_quant_dense(B, self.parameter_list, self.qnum, self.lookup, self.dropout, 
                                    expend_bit=self.up_bit, count=12)

            # Fully connected layer 3
            with tf.variable_scope('Exlogits', reuse=self.reuse):
                self.extend_fc_w8, self.extend_fc_w8_scale, self.hypothesis = \
                    util.extend_bit_log_quant_dense(B, self.parameter_list, self.qnum, self.lookup, self.dropout, 
                                    hypothesis=True, expend_bit=self.up_bit, count=14)

        return self.hypothesis

class AlexNet_binary:
    def __init__(self, X_img, FLAGS, dropout, block, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit

    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('layer1', reuse=self.reuse):
                self.conv_w1, self.conv_b1, B = util.binary_conv(B, 64, bias=0.0, c_s=2, pool=True)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('conv_w1', self.conv_w1)
            self.parameter_list.append(self.conv_w1)
            self.parameter_list.append(self.conv_b1)

            with tf.variable_scope('layer2', reuse=self.reuse):
                self.conv_w2, self.conv_b2, B = util.binary_conv(B, 192, bias=1.0, pool=True)
            self.parameter_list.append(self.conv_w2)
            self.parameter_list.append(self.conv_b2)

            with tf.variable_scope('layer3', reuse=self.reuse):
                self.conv_w3, self.conv_b3, B = util.binary_conv(B, 384, bias=1.0)
            self.parameter_list.append(self.conv_w3)
            self.parameter_list.append(self.conv_b3)

            with tf.variable_scope('layer4', reuse=self.reuse):
                self.conv_w4, self.conv_b4, B = util.binary_conv(B, 256, bias=1.0)
            self.parameter_list.append(self.conv_w4)
            self.parameter_list.append(self.conv_b4)

            with tf.variable_scope('layer5', reuse=self.reuse):
                self.conv_w5, self.conv_b5, B = util.binary_conv(B, 256, bias=1.0, pool=True, pd='SAME')
            self.parameter_list.append(self.conv_w5)
            self.parameter_list.append(self.conv_b5)

            with tf.variable_scope('layer6', reuse=self.reuse):
                self.fc_w6, self.fc_b6, B = util.binary_dense(B, 4096, self.dropout, fully=True)
            self.parameter_list.append(self.fc_w6)
            self.parameter_list.append(self.fc_b6)

            with tf.variable_scope('layer7', reuse=self.reuse):
                self.fc_w7, self.fc_b7, B = util.binary_dense(B, 4096, self.dropout)
            self.parameter_list.append(self.fc_w7)
            self.parameter_list.append(self.fc_b7)

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w8, self.fc_b8, self.hypothesis = util.binary_dense(B, self.num_class, self.dropout, hypothesis=True)
            self.parameter_list.append(self.fc_w8)
            self.parameter_list.append(self.fc_b8)
        return self.hypothesis
    
    def upgrade(self, X_img):
        B = X_img
        with tf.variable_scope('Train', reuse=self.reuse):
            with tf.variable_scope('Exlayer1', reuse=self.reuse):
                self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                    util.extend_bit_binary_conv(B, self.parameter_list, c_s=2, pool=True, count=0, expend_bit=self.up_bit)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w1', self.extend_conv_w1)

            # Convolution layer 2
            with tf.variable_scope('Exlayer2', reuse=self.reuse):
                self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                    util.extend_bit_binary_conv(B, self.parameter_list, pool=True, count=2, expend_bit=self.up_bit)
    
            # Convolution layer 3
            with tf.variable_scope('Exlayer3', reuse=self.reuse):
                self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                    util.extend_bit_binary_conv(B, self.parameter_list, count=4, expend_bit=self.up_bit)

            # Convolution layer 4
            with tf.variable_scope('Exlayer4', reuse=self.reuse):
                self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                    util.extend_bit_binary_conv(B, self.parameter_list, count=6, expend_bit=self.up_bit)

            # Convolution layer 5
            with tf.variable_scope('Exlayer5', reuse=self.reuse):
                self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                    util.extend_bit_binary_conv(B, self.parameter_list, pool=True, pd='SAME', count=8, expend_bit=self.up_bit)

            # Fully connected layer 1
            with tf.variable_scope('Exlayer6', reuse=self.reuse):
                self.extend_fc_w6, self.extend_fc_w6_scale, B = \
                    util.extend_bit_binary_dense(B, self.parameter_list, self.dropout, fully=True, count=10, expend_bit=self.up_bit)

            # Fully connected layer 2
            with tf.variable_scope('Exlayer7', reuse=self.reuse):
                self.extend_fc_w7, self.extend_fc_w7_scale, B = \
                    util.extend_bit_binary_dense(B, self.parameter_list, self.dropout, count=12, expend_bit=self.up_bit)

            # Fully connected layer 3
            with tf.variable_scope('Exlogits', reuse=self.reuse):
                self.extend_fc_w8, self.extend_fc_w8_scale, self.hypothesis = \
                    util.extend_bit_binary_dense(B, self.parameter_list, self.dropout, hypothesis=True, count=14, expend_bit=self.up_bit)

        return self.hypothesis

class VGG_linear:
    def __init__(self, X_img, FLAGS, block, dropout, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.block = block
        self.qnum = FLAGS.num_bit
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit
        
    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                       self.conv_w1, _,  B = util.conv(B, 64, self.qnum, pool=True, name='conv_w{}'.format(first))
                    else: 
                        self.conv_w1, _,  B = util.conv(B, 64, self.qnum, name='conv_w{}'.format(first))
                    self.parameter_list.append(self.conv_w1)
            with tf.device('/cpu:0'):
                tf.summary.histogram('conv_w1', self.conv_w1)

            with tf.variable_scope('block2', reuse=self.reuse):
                for second in range(self.block[1]):
                    if second == (self.block[1] - 1):
                       self.conv_w2, _,  B = util.conv(B, 128, self.qnum, pool=True, name='conv_w{}'.format(second))
                    else: 
                        self.conv_w2, _,  B = util.conv(B, 128, self.qnum, name='conv_w{}'.format(second))
                    self.parameter_list.append(self.conv_w2)

            with tf.variable_scope('block3', reuse=self.reuse):
                for third in range(self.block[2]):
                    if third == (self.block[2] - 1):
                       self.conv_w3, _,  B = util.conv(B, 256, self.qnum, k_s=1, pool=True, name='conv_w{}'.format(third))
                    else: 
                        self.conv_w3, _,  B = util.conv(B, 256, self.qnum, name='conv_w{}'.format(third), vg_drop=self.dropout)
                    self.parameter_list.append(self.conv_w3)

            with tf.variable_scope('block4', reuse=self.reuse):
                for fourth in range(self.block[3]):
                    if fourth == (self.block[3] - 1):
                       self.conv_w4, _,  B = util.conv(B, 512, self.qnum, k_s=1, pool=True, name='conv_w{}'.format(fourth))
                    else: 
                        self.conv_w4, _,  B = util.conv(B, 512, self.qnum, name='conv_w{}'.format(fourth), vg_drop=self.dropout)
                    self.parameter_list.append(self.conv_w4)

            with tf.variable_scope('block5', reuse=self.reuse):
                for fifth in range(self.block[4]):
                    if fifth == (self.block[4] - 1):
                       self.conv_w5, _,  B = util.conv(B, 512, self.qnum, k_s=1, pool=True, name='conv_w{}'.format(fifth))
                    else: 
                        self.conv_w5, _,  B = util.conv(B, 512, self.qnum, name='conv_w{}'.format(fifth))
                    self.parameter_list.append(self.conv_w5)

            with tf.variable_scope('FC6', reuse=self.reuse):
                self.fc_w14, self.fc_b14, L6 = util.vg_dense(B, 4096, self.qnum, self.dropout, fully=True)
            self.parameter_list.extend([self.fc_w14, self.fc_b14])

            with tf.variable_scope('FC7', reuse=self.reuse):
                self.fc_w15, self.fc_b15, L7 = util.vg_dense(L6, 4096, self.qnum, self.dropout)
            self.parameter_list.extend([self.fc_w15, self.fc_b15])

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w16, self.fc_b16, self.hypothesis = util.vg_dense(L7, self.num_class, self.qnum, self.dropout, hypothesis=True)
            self.parameter_list.extend([self.fc_w16, self.fc_b16])

        return self.hypothesis

    def upgrade(self, X_img):
        B = X_img
        with tf.variable_scope('Train', reuse=self.reuse):
            # Convolution layer 1
            with tf.variable_scope('ex_block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(first), pool=True, expend_bit=self.up_bit, count=first)
                    else:
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(first), expend_bit=self.up_bit, count=first)
                    self.upgarde_list.extend([self.extend_conv_w1, self.extend_conv_w1_scale])

                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w_a', self.extend_conv_w1)
                    # tf.summary.histogram('conv1_random_matrix', conv1_random_matrix)
                    # tf.summary.histogram('extend_conv_w1_scale', extend_conv_w1_scale)

            # Convolution layer 2
            with tf.variable_scope('ex_block2', reuse=self.reuse):
                start = self.block[0]
                end = start + self.block[1]
                for second in range(start, end):
                    if second == (end - 1):
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(second), pool=True, expend_bit=self.up_bit, count=second)
                    else:
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(second), expend_bit=self.up_bit, count=second) 
                    self.upgarde_list.extend([self.extend_conv_w2, self.extend_conv_w2_scale])

            # Convolution layer 3
            with tf.variable_scope('ex_block3', reuse=self.reuse):
                start = end
                end = start + self.block[2]
                for third in range(start, end):
                    if third == (end - 1):
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(third), pool=True, expend_bit=self.up_bit, count=third)
                    else:
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(third), expend_bit=self.up_bit, count=third) 
                    self.upgarde_list.extend([self.extend_conv_w3, self.extend_conv_w3_scale])

            # Convolution layer 4
            with tf.variable_scope('ex_block4', reuse=self.reuse):
                start = end
                end = start + self.block[3]
                for fourth in range(start, end):
                    if fourth == (end - 1):
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(fourth), pool=True, expend_bit=self.up_bit, count=fourth)
                    else:
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(fourth), expend_bit=self.up_bit, count=fourth) 
                    self.upgarde_list.extend([self.extend_conv_w4, self.extend_conv_w4_scale])

            # Convolution layer 5
            with tf.variable_scope('ex_block5', reuse=self.reuse):
                start = end
                end = start + self.block[4]
                for fifth in range(start, end):
                    if fifth == (end - 1):
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(fifth), pool=True, expend_bit=self.up_bit, count=fifth)
                    else:
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.vgg_e_conv(B, self.parameter_list, self.qnum, name='exconv_w{}'.format(fifth), expend_bit=self.up_bit, count=fifth) 
                    self.upgarde_list.extend([self.extend_conv_w5, self.extend_conv_w5_scale])

            # Fully connected layer 1
            with tf.variable_scope('ex_fc5', reuse=self.reuse):
                start = end
                self.extend_fc_w14, self.extend_fc_w14_scale, B6 = \
                    util.vgg_e_dense(B, self.parameter_list, self.qnum, self.dropout, count=start, expend_bit=self.up_bit, fully=True)
                self.upgarde_list.extend([self.extend_fc_w14, self.extend_fc_w14_scale])

            # Fully connected layer 2
            with tf.variable_scope('ex_fc6', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w15, self.extend_fc_w15_scale, B7 = \
                    util.vgg_e_dense(B6, self.parameter_list, self.qnum, self.dropout, expend_bit=self.up_bit, count=start)
                self.upgarde_list.extend([self.extend_fc_w15, self.extend_fc_w15_scale])

            # Fully connected layer 3
            with tf.variable_scope('ex_logits', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w16, self.extend_fc_w16_scale, self.hypothesis = \
                    util.vgg_e_dense(B7, self.parameter_list, self.qnum, self.dropout, count=start, expend_bit=self.up_bit, hypothesis=True)
                self.upgarde_list.extend([self.extend_fc_w16, self.extend_fc_w16_scale])

        return self.hypothesis


class VGG_log:
    def __init__(self, X_img, FLAGS, dropout, block, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.block = block
        self.qnum = FLAGS.num_bit
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit
        self.lookup = tf.linspace(0.0, -7.0, 2**self.qnum)

    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                       self.conv_w1, _,  B = util.log_conv(B, 64, self.qnum, self.lookup, pool=True, 
                                                name='conv_w{}'.format(first), dropout=self.dropout)
                    else: 
                        self.conv_w1, _,  B = util.log_conv(B, 64, self.qnum, self.lookup, name='conv_w{}'.format(first))
                    self.parameter_list.append(self.conv_w1)
            with tf.device('/cpu:0'):
                tf.summary.histogram('conv_w1', self.conv_w1)

            with tf.variable_scope('block2', reuse=self.reuse):
                for second in range(self.block[1]):
                    if second == (self.block[1] - 1):
                       self.conv_w2, _,  B = util.log_conv(B, 128, self.qnum, self.lookup, pool=True, 
                                                name='conv_w{}'.format(second), dropout=self.dropout)
                    else: 
                        self.conv_w2, _,  B = util.log_conv(B, 128, self.qnum, self.lookup, name='conv_w{}'.format(second))
                    self.parameter_list.append(self.conv_w2)

            with tf.variable_scope('block3', reuse=self.reuse):
                for third in range(self.block[2]):
                    if third == (self.block[2] - 1):
                       self.conv_w3, _,  B = util.log_conv(B, 256, self.qnum, self.lookup, k_s=1, pool=True, 
                                                name='conv_w{}'.format(third))
                    else: 
                        self.conv_w3, _,  B = util.log_conv(B, 256, self.qnum, self.lookup, dropout=self.dropout, name='conv_w{}'.format(third))
                    self.parameter_list.append(self.conv_w3)

            with tf.variable_scope('block4', reuse=self.reuse):
                for fourth in range(self.block[3]):
                    if fourth == (self.block[3] - 1):
                       self.conv_w4, _,  B = util.log_conv(B, 512, self.qnum, self.lookup, k_s=1, pool=True, 
                                                name='conv_w{}'.format(fourth))
                    else: 
                        self.conv_w4, _,  B = util.log_conv(B, 512, self.qnum, self.lookup, dropout=self.dropout, name='conv_w{}'.format(fourth))
                    self.parameter_list.append(self.conv_w4)

            with tf.variable_scope('block5', reuse=self.reuse):
                for fifth in range(self.block[4]):
                    if fifth == (self.block[4] - 1):
                       self.conv_w5, _,  B = util.log_conv(B, 512, self.qnum, self.lookup, k_s=1, pool=True, 
                                                name='conv_w{}'.format(fifth))
                    else: 
                        self.conv_w5, _,  B = util.log_conv(B, 512, self.qnum, self.lookup, dropout=self.dropout, name='conv_w{}'.format(fifth))
                    self.parameter_list.append(self.conv_w5)

            with tf.variable_scope('FC6', reuse=self.reuse):
                self.fc_w14, self.fc_b14, L6 = util.log_dense(B, 4096, self.qnum, self.lookup, self.dropout, fully=True)
            self.parameter_list.extend([self.fc_w14, self.fc_b14])

            with tf.variable_scope('FC7', reuse=self.reuse):
                self.fc_w15, self.fc_b15, L7 = util.log_dense(L6, 4096, self.qnum, self.lookup, self.dropout)
            self.parameter_list.extend([self.fc_w15, self.fc_b15])

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w16, self.fc_b16, self.hypothesis = util.log_dense(L7, self.num_class, self.qnum,self.lookup, self.dropout, hypothesis=True)
            self.parameter_list.extend([self.fc_w16, self.fc_b16])

        return self.hypothesis

    def upgrade(self, X_img):
        B = X_img  
        with tf.variable_scope('Train', reuse=self.reuse):
            # Convolution layer 1
            with tf.variable_scope('ex_block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(first), pool=True, count=first,  expend_bit=self.up_bit, dropout=self.dropout)
                    else:
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(first),  expend_bit=self.up_bit, count=first)
                    self.upgarde_list.extend([self.extend_conv_w1, self.extend_conv_w1_scale])

                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w_a', self.extend_conv_w1)
                    # tf.summary.histogram('conv1_random_matrix', conv1_random_matrix)
                    # tf.summary.histogram('extend_conv_w1_scale', extend_conv_w1_scale)

            # Convolution layer 2
            with tf.variable_scope('ex_block2', reuse=self.reuse):
                start = self.block[0]
                end = start + self.block[1]
                for second in range(start, end):
                    if second == (end - 1):
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(second), pool=True, count=second, expend_bit=self.up_bit, dropout=self.dropout)
                    else:
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(second), expend_bit=self.up_bit, count=second) 
                    self.upgarde_list.extend([self.extend_conv_w2, self.extend_conv_w2_scale])

            # Convolution layer 3
            with tf.variable_scope('ex_block3', reuse=self.reuse):
                start = end
                end = start + self.block[2]
                for third in range(start, end):
                    if third == (end - 1):
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(third), pool=True, count=third, expend_bit=self.up_bit, dropout=self.dropout)
                    else:
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(third), expend_bit=self.up_bit, count=third) 
                    self.upgarde_list.extend([self.extend_conv_w3, self.extend_conv_w3_scale])

            # Convolution layer 4
            with tf.variable_scope('ex_block4', reuse=self.reuse):
                start = end
                end = start + self.block[3]
                for fourth in range(start, end):
                    if fourth == (end - 1):
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(fourth), pool=True, count=fourth, expend_bit=self.up_bit, dropout=self.dropout)
                    else:
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(fourth), expend_bit=self.up_bit, count=fourth) 
                    self.upgarde_list.extend([self.extend_conv_w4, self.extend_conv_w4_scale])

            # Convolution layer 5
            with tf.variable_scope('ex_block5', reuse=self.reuse):
                start = end
                end = start + self.block[4]
                for fifth in range(start, end):
                    if fifth == (end - 1):
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(fifth), pool=True, count=fifth, expend_bit=self.up_bit, dropout=self.dropout)
                    else:
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.vgg_log_e_conv(B, self.parameter_list, self.qnum, self.lookup, 
                                        name='exconv_w{}'.format(fifth), expend_bit=self.up_bit, count=fifth) 
                    self.upgarde_list.extend([self.extend_conv_w5, self.extend_conv_w5_scale])

            # Fully connected layer 1
            with tf.variable_scope('ex_fc5', reuse=self.reuse):
                start = end
                self.extend_fc_w14, self.extend_fc_w14_scale, B6 = \
                    util.vgg_log_e_dense(B, self.parameter_list, self.qnum, self.lookup, dropout=self.dropout, 
                                count=start, expend_bit=self.up_bit, fully=True)
                self.upgarde_list.extend([self.extend_fc_w14, self.extend_fc_w14_scale])

            # Fully connected layer 2
            with tf.variable_scope('ex_fc6', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w15, self.extend_fc_w15_scale, B7 = \
                    util.vgg_log_e_dense(B6, self.parameter_list, self.qnum, self.lookup, dropout=self.dropout, 
                                expend_bit=self.up_bit, count=start)
                self.upgarde_list.extend([self.extend_fc_w15, self.extend_fc_w15_scale])

            # Fully connected layer 3
            with tf.variable_scope('ex_logits', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w16, self.extend_fc_w16_scale, self.hypothesis = \
                    util.vgg_log_e_dense(B7, self.parameter_list, self.qnum, self.lookup, dropout=self.dropout, 
                                expend_bit=self.up_bit, count=start, hypothesis=True)
                self.upgarde_list.extend([self.extend_fc_w16, self.extend_fc_w16_scale])

        return self.hypothesis

class VGG_binary:
    def __init__(self, X_img, FLAGS, dropout, block, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.block = block
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit

    def original(self, X_img):
        B = X_img
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                       self.conv_w1, _,  B = util.binary_conv(B, 64, pool=True, name='conv_w{}'.format(first))
                    else: 
                        self.conv_w1, _,  B = util.binary_conv(B, 64, name='conv_w{}'.format(first))
                    self.parameter_list.append(self.conv_w1)
            with tf.device('/cpu:0'):
                tf.summary.histogram('conv_w1', self.conv_w1)

            with tf.variable_scope('block2', reuse=self.reuse):
                for second in range(self.block[1]):
                    if second == (self.block[1] - 1):
                       self.conv_w2, _,  B = util.binary_conv(B, 128, pool=True, name='conv_w{}'.format(second))
                    else: 
                        self.conv_w2, _,  B = util.binary_conv(B, 128, name='conv_w{}'.format(second))
                    self.parameter_list.append(self.conv_w2)

            with tf.variable_scope('block3', reuse=self.reuse):
                for third in range(self.block[2]):
                    if third == (self.block[2] - 1):
                       self.conv_w3, _,  B = util.binary_conv(B, 256, k_s=1, pool=True, name='conv_w{}'.format(third))
                    else: 
                        self.conv_w3, _,  B = util.binary_conv(B, 256, name='conv_w{}'.format(third))
                    self.parameter_list.append(self.conv_w3)

            with tf.variable_scope('block4', reuse=self.reuse):
                for fourth in range(self.block[3]):
                    if fourth == (self.block[3] - 1):
                       self.conv_w4, _,  B = util.binary_conv(B, 512, k_s=1, pool=True, name='conv_w{}'.format(fourth))
                    else: 
                        self.conv_w4, _,  B = util.binary_conv(B, 512, name='conv_w{}'.format(fourth))
                    self.parameter_list.append(self.conv_w4)

            with tf.variable_scope('block5', reuse=self.reuse):
                for fifth in range(self.block[4]):
                    if fifth == (self.block[4] - 1):
                       self.conv_w5, _,  B = util.binary_conv(B, 512, k_s=1, pool=True, name='conv_w{}'.format(fifth))
                    else: 
                        self.conv_w5, _,  B = util.binary_conv(B, 512, name='conv_w{}'.format(fifth))
                    self.parameter_list.append(self.conv_w5)

            with tf.variable_scope('FC6', reuse=self.reuse):
                self.fc_w14, self.fc_b14, L6 = util.vg_binary_dense(B, 4096, self.dropout, fully=True)
            self.parameter_list.extend([self.fc_w14, self.fc_b14])

            with tf.variable_scope('FC7', reuse=self.reuse):
                self.fc_w15, self.fc_b15, L7 = util.vg_binary_dense(L6, 4096, self.dropout)
            self.parameter_list.extend([self.fc_w15, self.fc_b15])

            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w16, self.fc_b16, self.hypothesis = util.vg_binary_dense(L7, self.num_class, dropout=self.dropout, hypothesis=True)
            self.parameter_list.extend([self.fc_w16, self.fc_b16])
        
        return self.hypothesis

    def upgrade(self, X_img):
        B = X_img
        with tf.variable_scope('Train', reuse=self.reuse):
            # Convolution layer 1
            with tf.variable_scope('ex_block1', reuse=self.reuse):
                for first in range(self.block[0]):
                    if first == (self.block[0] - 1):
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(first), 
                                pool=True, expend_bit=self.up_bit, count=first)
                    else:
                        self.extend_conv_w1, self.extend_conv_w1_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(first), 
                                expend_bit=self.up_bit, count=first)
                    self.upgarde_list.extend([self.extend_conv_w1, self.extend_conv_w1_scale])

                with tf.device('/cpu:0'):
                    tf.summary.histogram('extend_conv_w_a', self.extend_conv_w1)
                    # tf.summary.histogram('conv1_random_matrix', conv1_random_matrix)
                    # tf.summary.histogram('extend_conv_w1_scale', extend_conv_w1_scale)

            # Convolution layer 2
            with tf.variable_scope('ex_block2', reuse=self.reuse):
                start = self.block[0]
                end = start + self.block[1]
                for second in range(start, end):
                    if second == (end - 1):
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(second), 
                                pool=True, expend_bit=self.up_bit, count=second)
                    else:
                        self.extend_conv_w2, self.extend_conv_w2_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None,name='exconv_w{}'.format(second), 
                                expend_bit=self.up_bit, count=second) 
                    self.upgarde_list.extend([self.extend_conv_w2, self.extend_conv_w2_scale])

            # Convolution layer 3
            with tf.variable_scope('ex_block3', reuse=self.reuse):
                start = end
                end = start + self.block[2]
                for third in range(start, end):
                    if third == (end - 1):
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(third), 
                                pool=True, expend_bit=self.up_bit, count=third)
                    else:
                        self.extend_conv_w3, self.extend_conv_w3_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(third), 
                                expend_bit=self.up_bit, count=third) 
                    self.upgarde_list.extend([self.extend_conv_w3, self.extend_conv_w3_scale])

            # Convolution layer 4
            with tf.variable_scope('ex_block4', reuse=self.reuse):
                start = end
                end = start + self.block[3]
                for fourth in range(start, end):
                    if fourth == (end - 1):
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(fourth), 
                                pool=True, expend_bit=self.up_bit, count=fourth)
                    else:
                        self.extend_conv_w4, self.extend_conv_w4_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(fourth), 
                                expend_bit=self.up_bit, count=fourth) 
                    self.upgarde_list.extend([self.extend_conv_w4, self.extend_conv_w4_scale])

            # Convolution layer 5
            with tf.variable_scope('ex_block5', reuse=self.reuse):
                start = end
                end = start + self.block[4]
                for fifth in range(start, end):
                    if fifth == (end - 1):
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(fifth), 
                                pool=True, expend_bit=self.up_bit, count=fifth)
                    else:
                        self.extend_conv_w5, self.extend_conv_w5_scale, B = \
                            util.extend_bit_binary_conv(B, self.parameter_list, bias=None, name='exconv_w{}'.format(fifth), 
                                expend_bit=self.up_bit, count=fifth) 
                    self.upgarde_list.extend([self.extend_conv_w5, self.extend_conv_w5_scale])

            # Fully connected layer 1
            with tf.variable_scope('ex_fc5', reuse=self.reuse):
                start = end
                self.extend_fc_w14, self.extend_fc_w14_scale, B6 = \
                    util.extend_bit_binary_dense(B, self.parameter_list, self.dropout, fully=True, 
                        expend_bit=self.up_bit, count=start)
                self.upgarde_list.extend([self.extend_fc_w14, self.extend_fc_w14_scale])

            # Fully connected layer 2
            with tf.variable_scope('ex_fc6', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w15, self.extend_fc_w15_scale, B7 = \
                    util.extend_bit_binary_dense(B6, self.parameter_list, self.dropout, 
                        expend_bit=self.up_bit, count=start)
                self.upgarde_list.extend([self.extend_fc_w15, self.extend_fc_w15_scale])

            # Fully connected layer 3
            with tf.variable_scope('ex_logits', reuse=self.reuse):
                start = start + 2
                self.extend_fc_w16, self.extend_fc_w16_scale, self.hypothesis = \
                    util.extend_bit_binary_dense(B7, self.parameter_list, self.dropout, 
                        expend_bit=self.up_bit, count=start, hypothesis=True)
                self.upgarde_list.extend([self.extend_fc_w16, self.extend_fc_w16_scale])

        return self.hypothesis

class ResNet20_linear:
    def __init__(self, X_img, FLAGS, block, dropout, num_class=100, reuse=False):
        self.num_class = num_class
        self.parameter_list = []
        self.upgarde_list = []
        self.qnum = FLAGS.num_bit
        self.reuse = reuse
        self.dropout = dropout
        self.up_bit = FLAGS.up_bit

    def original(self, X_img):
        with tf.variable_scope('Model', reuse=self.reuse):
            with tf.variable_scope('conv1', reuse=self.reuse):
                self.conv_w1, _, L = util.conv(X_img, 16, self.qnum, c_s=1, q=False)
                self.parameter_list.append(self.conv_w1)
                with tf.device('/cpu:0'):
                    tf.summary.histogram('conv_w1', self.conv_w1)

            with tf.variable_scope('conv2_x', reuse=self.reuse):
                self.conv2_a, self.conv2_b, self.conv2_s, L = util.residual_first_2d(L, [16, 16], self.qnum)
                self.parameter_list.extend([self.conv2_a, self.conv2_b, self.conv2_s])
                for second_block in ['1b_', '1c_']:
                    self.conv2_block_a, self.conv2_block_b, self.conv2_block_s, L = \
                            util.residual_first_2d(L, [16, 16], self.qnum, block='block_2_{}'.format(second_block))
                    self.parameter_list.extend([self.conv2_block_a, self.conv2_block_b, self.conv2_block_s])

            with tf.variable_scope('conv3_x', reuse=self.reuse):
                self.conv3_a, self.conv3_b, self.conv3_s, L = util.residual_first_2d(L, [32, 32], self.qnum, first=True)
                self.parameter_list.extend([self.conv3_a, self.conv3_b, self.conv3_s])
                for third_block in ['1b_', '1c_']:
                    self.conv3_block_a, self.conv3_block_b, self.conv3_block_s, L = \
                            util.residual_first_2d(L, [32, 32], self.qnum, block='block_3_{}'.format(third_block))
                    self.parameter_list.extend([self.conv3_block_a, self.conv3_block_b, self.conv3_block_s])
            
            with tf.variable_scope('conv4_x', reuse=self.reuse):
                self.conv4_a, self.conv4_b, self.conv4_s, L = util.residual_first_2d(L, [64, 64], self.qnum, first=True)
                self.parameter_list.extend([self.conv4_a, self.conv4_b, self.conv4_s])
                for fourth_block in ['1b_', '1c_']:
                    self.conv4_block_a, self.conv4_block_b, self.conv4_block_s, L = \
                            util.residual_first_2d(L, [64, 64], self.qnum, block='block_4_{}'.format(fourth_block))
                    self.parameter_list.extend([self.conv4_block_a, self.conv4_block_b, self.conv4_block_s])

            with tf.variable_scope('pool', reuse=self.reuse):
                L = tf.reduce_mean(L, axis=[1,2])
            with tf.variable_scope('logits', reuse=self.reuse):
                self.fc_w, self.fc_b, self.hypothesis = util.dense(L, self.num_class, self.qnum, dropout=self.dropout, hypothesis=True, q=False)
                self.parameter_list.extend([self.fc_w, self.fc_b])
        return self.hypothesis

