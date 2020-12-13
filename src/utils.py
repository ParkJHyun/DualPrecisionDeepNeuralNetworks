import tensorflow as tf
import quant_utils as qu

def conv(input_, output_channel, qnum, bias=None, c_s=1, k_s=3, pool=False, p_k=2, p_s=2, pd='VALID', act=True, name='conv_w', vg_drop=None, q=True):
    input_shape = input_.get_shape().as_list()
    input_shape = input_shape[3]
    conv_w = tf.get_variable(name=name, shape=[k_s, k_s, input_shape, output_channel], 
                                initializer=tf.contrib.layers.variance_scaling_initializer())
    if bias is not None:
        conv_b = tf.get_variable(name='conv_b', shape=[output_channel], 
                                initializer=tf.constant_initializer(bias))
    else:
        conv_b = None

    # Quantization Training
    if q:
        q_conv_w, _ = qu._quantize_scale_print(conv_w, qnum=qnum)
        conv_w = conv_w + tf.stop_gradient(q_conv_w - conv_w)

    convs = tf.nn.conv2d(input_, conv_w, strides=[1,c_s,c_s,1], padding='SAME', name='conv')
    if bias is not None:
        convs = tf.nn.bias_add(convs, conv_b, name='preact')
    L = tf.contrib.layers.batch_norm(convs, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L, name='Activation')

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd, name='pool')
    if vg_drop is not None:
        L = tf.nn.dropout(L, keep_prob=vg_drop, name='dropout')
        return conv_w, conv_b, L
    return conv_w, conv_b, L

def dense(input_, output_channel, qnum, dropout, fully=False, hypothesis=False, q=True):
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
        fc_w = tf.get_variable(name='fc_w', shape=[flat_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())
    else:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]
        fc_w = tf.get_variable(name='fc_w', shape=[input_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())

    fc_b = tf.get_variable(name='fc_b', shape=[output_channel], initializer=tf.constant_initializer(0.0))
    
    # Quantization Training
    if q:
        q_fc_w, _ = qu._quantize_scale_print(fc_w, qnum=qnum)
        fc_w = fc_w + tf.stop_gradient(q_fc_w - fc_w)

    fc_ = tf.matmul(input_, fc_w)
    fc_ = fc_ + fc_b
    if hypothesis:
        return fc_w, fc_b, fc_
    else:
        L1 = tf.contrib.layers.batch_norm(fc_, scale=True, is_training=True)
        L1 = tf.nn.relu(L1, name='Relu')
        if dropout is not None:
            L1 = tf.nn.dropout(L1, keep_prob=dropout, name='dropout')
        return fc_w, fc_b, L1

def vg_dense(input_, output_channel, qnum, dropout=None, fully=False, hypothesis=False):
    fc_w, fc_b, fc_ = dense(input_, output_channel, qnum, dropout=dropout, fully=fully, hypothesis=hypothesis)
    return fc_w, fc_b, fc_

def extend_bit_quant_conv(input_, param_lst, qnum, bias=True, c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', dropout=None, count=0, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    if bias:
        input_b = param_lst[tmp+1]
    else:
        input_b = None
    Q_conv_w, Q_conv_w_s = qu.class_quantize(input_w, quantize_num=qnum)
    conv_random_matrix = tf.get_variable(name='conv_random', shape=Q_conv_w.get_shape(), initializer=tf.constant_initializer(0.0))
    # conv_random_matrix = tf.Variable(tf.random_normal(shape=Q_conv_w.get_shape(), mean=0.0, stddev=0.002), 'conv_random')
    extend_conv_w, extend_conv_w_scale = qu._quant_extend_bit(quant_w=Q_conv_w, scale=Q_conv_w_s, 
                random_matrix=conv_random_matrix, name='new_conv_w', qnum=qnum, expend_bit=expend_bit)
    L = qu.conv_2d(inputs=input_, level=extend_conv_w, bias=input_b, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w') 

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')
        return extend_conv_w, extend_conv_w_scale, L

    return extend_conv_w, extend_conv_w_scale, L

def extend_bit_quant_dense(input_, param_lst, qnum, dropout, fully=False, hypothesis=False, count=10, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]
    
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
    Q_fc_w, Q_fc_w_s = qu.class_quantize(input_w, quantize_num=qnum)
    fc_random_matrix = tf.get_variable(name='fc_random', shape=Q_fc_w.get_shape(), 
                                     initializer=tf.constant_initializer(0.0))
    # fc_random_matrix = tf.Variable(tf.random_normal(shape=Q_fc_w.get_shape(), mean=0.0, stddev=0.002), name='fc_random')
    extend_fc_w, extend_fc_w_scale = qu._quant_extend_bit(quant_w=Q_fc_w, scale=Q_fc_w_s, 
        random_matrix=fc_random_matrix, name='new_fc_w', qnum=qnum, expend_bit=expend_bit)
    if hypothesis:
        L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='', name='dequant_fc_w')
        return extend_fc_w, extend_fc_w_scale, L
    L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='relu', name='dequant_fc_w')
    L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L

def vgg_e_conv(input_, param_list, qnum, name, c_s=1, act=True, pool=False, p_k=2, p_s=2, pd='VALID', count=0, dropout=None, expend_bit=1):
    scale_rate = 1
    expend_bit = expend_bit
    input_w = param_list[count]
    Q_conv_w, Q_conv_w_s = qu.class_quantize(input_w, quantize_num=qnum)
    conv_random_matrix = tf.Variable(tf.random_normal(shape=Q_conv_w.get_shape(), mean=0.0, stddev=0.3), name='conv_random')
    # conv_random_matrix = tf.get_variable(name='conv_random_'+name, shape=Q_conv_w.get_shape(), 
    #                                         initializer=tf.constant_initializer(0.0))
    extend_conv_w, extend_conv_w_scale = qu._quant_extend_bit(quant_w=Q_conv_w, scale=Q_conv_w_s, 
                random_matrix=conv_random_matrix, expend_bit=expend_bit, name='new_conv_w_'+name, scale_rate=scale_rate, qnum=qnum)
    L = qu.only_conv_2d(inputs=input_, level=extend_conv_w, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w_'+name)
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L)

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        if dropout is not None:
            L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_conv_w, extend_conv_w_scale, L

def vgg_e_dense(input_, param_lst, qnum, dropout=None, count=13, fully=False, hypothesis=False, expend_bit=1):
    scale_rate = 1
    expend_bit = expend_bit

    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]

    if fully:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, input_shape])
    quant_fc_w, quant_fc_w_scale = qu.class_quantize(input_w, quantize_num=qnum)
    fc_random_matrix = tf.Variable(tf.random_normal(shape=quant_fc_w.get_shape(), mean=0.0, stddev=0.3), name='fc6_random')
    # fc_random_matrix = tf.get_variable(name='fc_random', shape=quant_fc_w.get_shape(), 
    #                                     initializer=tf.constant_initializer(0.0))  
    extend_fc_w, extend_fc_w_scale = qu._quant_extend_bit(quant_w=quant_fc_w, scale=quant_fc_w_scale, 
        random_matrix=fc_random_matrix, expend_bit=expend_bit, name='new_fc_w', scale_rate=scale_rate, qnum=qnum)
    L = qu.only_denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, name='dequant_fc_w')

    if hypothesis:
        return extend_fc_w, extend_fc_w_scale, L
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L

def conv_block_2d(input_, filters, qnum, block='', small=None):
    if small is not None:
        output_ch1, output_ch2 = filters

        with tf.variable_scope(block + 'conv_a'):
            conv_wa, _, B1 = conv(input_, output_ch1, qnum, c_s=1)
        with tf.variable_scope(block + 'conv_b'):
            conv_wb, _, B2 = conv(B1, output_ch2, qnum, c_s=1, act=False)
        with tf.variable_scope(block + 'shortcut'):
            shortcut, _, B3 = conv(input_, output_ch2, qnum, c_s=1, act=False)

        B4 = tf.add(B3, B2)
        B4 = tf.nn.relu(B4)
        return conv_wa, conv_wb, shortcut, B4
    
    output_ch1, output_ch2, output_ch3 = filters
    with tf.variable_scope(block + 'conv_a'):
        conv_wa, _, B1 = conv(input_, output_ch1, qnum, k_s=1)
    with tf.variable_scope(block + 'conv_b'):
        conv_wb, _, B2 = conv(B1, output_ch2, qnum, k_s=3)
    with tf.variable_scope(block + 'conv_c'):
        conv_wc, _, B3 = conv(B2, output_ch3, qnum, k_s=1)
    with tf.variable_scope(block + 'shortcut'):
        shortcut, _, B4 = conv(input_, output_ch3, qnum, k_s=1, act=False)

    B5 = tf.add(B4, B3)
    B5 = tf.nn.relu(B5)

    return conv_wa, conv_wb, conv_wc, shortcut, B5

def identity_block_2d(input_, filters, qnum, block, small=None):
    if small is not None:
        output_ch1, output_ch2 = filters
        with tf.variable_scope(block + 'conv_a'):
            conv_wa, _, B1 = conv(input_, output_ch1, qnum, k_s=3)
        with tf.variable_scope(block + 'conv_b'):
            conv_wb, _, B2 = conv(B1, output_ch2, qnum, k_s=3, act=False)
        B3 = tf.add(B2, input_)
        B3 = tf.nn.relu(B3)
        return conv_wa, conv_wb, B3
        
    output_ch1, output_ch2, output_ch3 = filters
    with tf.variable_scope(block + 'conv_a'):
        conv_wa, _, B1 = conv(input_, output_ch1, qnum, k_s=1)
    with tf.variable_scope(block + 'conv_b'):
        conv_wb, _, B2 = conv(B1, output_ch2, qnum, k_s=3)
    with tf.variable_scope(block + 'conv_c'):
        conv_wc, _, B3 = conv(B2, output_ch3, qnum, k_s=1, act=False)
    
    B4 = tf.add(B3, input_)
    B4 = tf.nn.relu(B4)

    return conv_wa, conv_wb, conv_wc, B4

def residual_unit_2d(input_, filters, qnum, block):
    output_ch1, output_ch2 = filters
    with tf.variable_scope(block + 'conv_a'):
        conv_wa, _, B1 = conv(input_, output_ch1, qnum, k_s=3)
    with tf.variable_scope(block + 'conv_b'):
        conv_wb, _, B2 = conv(B1, output_ch2, qnum, k_s=3, act=False)
    B3 = tf.add(B2, input_)
    B3 = tf.nn.relu(B3)
    return conv_wa, conv_wb, B3

def residual_first_2d(input_, filters, qnum, block='', first=False):
    output_ch1, output_ch2 = filters

    if first:
        with tf.variable_scope(block + 'conv_a'):
            conv_wa, _, B1 = conv(input_, output_ch1, qnum, c_s=2)
        with tf.variable_scope(block + 'conv_b'):
            conv_wb, _, B2 = conv(B1, output_ch2, qnum, c_s=1, act=False)
        with tf.variable_scope(block + 'shortcut'):
            shortcut, _, B3 = conv(input_, output_ch2, qnum, c_s=2, act=False)

    else :
        with tf.variable_scope(block + 'conv_a'):
            conv_wa, _, B1 = conv(input_, output_ch1, qnum, c_s=1)
        with tf.variable_scope(block + 'conv_b'):
            conv_wb, _, B2 = conv(B1, output_ch2, qnum, c_s=1, act=False)
        with tf.variable_scope(block + 'shortcut'):
            shortcut, _, B3 = conv(input_, output_ch2, qnum, c_s=1, act=False)

    B4 = tf.add(B3, B2)
    B4 = tf.nn.relu(B4)
    L4, _ = qu._quantize_scale_print(B4, qnum=qnum)
    B4 = B4 + tf.stop_gradient(L4 - B4)
    return conv_wa, conv_wb, shortcut, B4


def log_conv(input_, output_channel, qnum, lookup, bias=None, c_s=1, k_s=3, pool=False, p_k=2, p_s=2, pd='VALID', name='log_conv_w', dropout=None):
    input_shape = input_.get_shape().as_list()
    input_shape = input_shape[3]
    conv_w = tf.get_variable(name=name, shape=[k_s, k_s, input_shape, output_channel],  
                                initializer=tf.contrib.layers.xavier_initializer())
    if bias is not None:
        conv_b = tf.get_variable(name='conv_b', shape=[output_channel], 
                                initializer=tf.constant_initializer(bias))
    else:
        conv_b = None

    # Quantization Training
    q_conv_w = qu._log_quant2(conv_w, qnum=qnum, lookup=lookup)
    conv_w = conv_w + tf.stop_gradient(q_conv_w - conv_w)

    convs = tf.nn.conv2d(input_, conv_w, strides=[1,c_s,c_s,1], padding='SAME', name='conv')
    if bias is not None:
        convs = tf.nn.bias_add(convs, conv_b, name='preact')
    L = tf.contrib.layers.batch_norm(convs, scale=True, is_training=True)
    L = tf.nn.relu(L, name='Activation')

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd, name='pool')
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')
    return conv_w, conv_b, L

def log_dense(input_, output_channel, qnum, lookup, dropout, fully=False, hypothesis=False):
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
        fc_w = tf.get_variable(name='fc_w', shape=[flat_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())
    else:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]
        fc_w = tf.get_variable(name='fc_w', shape=[input_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())

    fc_b = tf.get_variable(name='fc_b', shape=[output_channel], initializer=tf.constant_initializer(0.0))
    
    # Quantization Training
    q_fc_w = qu._log_quant2(fc_w, qnum=qnum, lookup=lookup, fc=True)
    fc_w = fc_w + tf.stop_gradient(q_fc_w - fc_w)

    fc_ = tf.matmul(input_, fc_w)
    fc_ = fc_ + fc_b
    if hypothesis:
        return fc_w, fc_b, fc_
    else:
        L1 = tf.contrib.layers.batch_norm(fc_, scale=True, is_training=True)
        L1 = tf.nn.relu(L1, name='Relu')
        if dropout is not None:
            L1 = tf.nn.dropout(L1, keep_prob=dropout, name='dropout')
        return fc_w, fc_b, L1

def extend_bit_log_quant_conv(input_, param_lst, qnum, lookup, c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', count=0, expend_bit=1, dropout=None):
    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]

    conv_random_matrix = tf.get_variable(name='conv_random', shape=input_w.get_shape(), initializer=tf.constant_initializer(0.0))
    level, scale, Q_conv_w = qu.extend_log_quantize(input_w, lookup, conv_random_matrix, quantize_num=qnum, expend_bit=expend_bit)
    L = tf.nn.conv2d(input_, Q_conv_w, strides=[1, c_s, c_s, 1], padding='SAME', name='dequant_conv_w')
    if input_b is not None:
        L = tf.nn.bias_add(L, input_b)
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        return level, scale, L
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return level, scale, L

def extend_bit_log_quant_dense(input_, param_lst, qnum, lookup, dropout, fully=False, hypothesis=False, count=10, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]
    
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
    fc_random_matrix = tf.get_variable(name='fc_random', shape=input_w.get_shape(), 
                                        initializer=tf.constant_initializer(0.0))
    level, scale, Q_conv_w = qu.extend_log_quantize(input_w, lookup, fc_random_matrix, quantize_num=qnum, dense=True, expend_bit=expend_bit)
    L = tf.matmul(input_, Q_conv_w)
    L = L + input_b

    if hypothesis:
        return level, scale, L

    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)
    L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return level, scale, L

def ex_conv_block_2d(input_, param_list, qnum, count=0, small=None):
    tmp = count

    if small is not None:
        extend_conv_wa, _, L = vgg_e_conv(input_, param_list, qnum, name='a', count=tmp)
        extend_conv_wb, _, L = vgg_e_conv(L, param_list, qnum, name='b', count=tmp+1)
        extend_conv_shrcut, _, L1 = vgg_e_conv(input_, param_list, qnum, name='shortcut', act=False, count=tmp+2)

        L = tf.add(L, L1)
        L = tf.nn.relu(L)

        return extend_conv_wa, extend_conv_wb, extend_conv_shrcut, L 

    extend_conv_wa, _, L = vgg_e_conv(input_, param_list, qnum, name='a', count=tmp)
    extend_conv_wb, _, L1 = vgg_e_conv(L, param_list, qnum, name='b', count=tmp+1)
    extend_conv_wc, _, L2 = vgg_e_conv(L1, param_list, qnum, name='c', count=tmp+2)
    extend_conv_shrcut, _, L3 = vgg_e_conv(input_, param_list, qnum, name='shortcut', act=False, count=tmp+3)

    L = tf.add(L3, L2)
    L = tf.nn.relu(L)

    return extend_conv_wa, extend_conv_wb, extend_conv_wc, extend_conv_shrcut, L

def ex_identity_block_2d(input_, param_list, qnum, block, count=0, small=None):
    tmp = count

    if small is not None:
        with tf.variable_scope(block + 'ex_conv_a'):
            ex_conv_wa, _, L = vgg_e_conv(input_, param_list, qnum, name='a', count=tmp)
        with tf.variable_scope(block + 'ex_conv_b'):
            ex_conv_wb, _, L1 = vgg_e_conv(L, param_list, qnum, name='b', act=False, count=tmp+1)

        L = tf.add(L1, input_)
        L = tf.nn.relu(L)

        return ex_conv_wa, ex_conv_wb, L

    with tf.variable_scope(block + 'ex_conv_a'):
        ex_conv_wa, _, L = vgg_e_conv(input_, param_list, qnum, name='a', count=tmp)
    with tf.variable_scope(block + 'ex_conv_b'):
        ex_conv_wb, _, L1 = vgg_e_conv(L, param_list, qnum, name='b', count=tmp+1)
    with tf.variable_scope(block + 'ex_conv_c'):
        ex_conv_wc, _, L2 = vgg_e_conv(L1, param_list, qnum, name='c', act=False, count=tmp+2)
    
    L = tf.add(L2, input_)
    L = tf.nn.relu(L)

    return ex_conv_wa, ex_conv_wb, ex_conv_wc, L

def vgg_log_e_conv(input_, param_list, qnum, lookup, name, c_s=1, act=True, pool=False, p_k=2, p_s=2, pd='VALID', count=0, dropout=None, expend_bit=1):
    input_w = param_list[count]
    conv_random_matrix = tf.get_variable(name='conv_random_'+name, shape=input_w.get_shape(), 
                                            initializer=tf.constant_initializer(0.0))
    level, scale, Q_conv_w = qu.extend_log_quantize(input_w, lookup, conv_random_matrix, expend_bit=expend_bit,quantize_num=qnum)
    L = tf.nn.conv2d(input_, Q_conv_w, strides=[1, c_s, c_s, 1], padding='SAME', name='dequant_conv_w')
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L)

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        if dropout is not None:
            L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')
        return level, scale, L

    return level, scale, L

def vgg_log_e_dense(input_, param_lst, qnum, lookup, dropout=None, fully=False, hypothesis=False, count=10, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]

    if fully:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, input_shape])
    fc_random_matrix = tf.get_variable(name='fc_random', shape=input_w.get_shape(), 
                                        initializer=tf.constant_initializer(0.0))
    level, scale, Q_conv_w = qu.extend_log_quantize(input_w, lookup, fc_random_matrix, expend_bit=expend_bit, quantize_num=qnum, dense=True)
    L = tf.matmul(input_, Q_conv_w)
    L = L + input_b

    if hypothesis:
        return level, scale, L
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return level, scale, L

def binary_conv(input_, output_channel, bias=None, c_s=1, k_s=3, pool=False, p_k=2, p_s=2, pd='VALID', act=True, name='conv_w'):
    input_shape = input_.get_shape().as_list()
    input_shape = input_shape[3]
    conv_w = tf.get_variable(name=name, shape=[k_s, k_s, input_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())
    if bias is not None:
        conv_b = tf.get_variable(name='conv_b', shape=[output_channel], 
                                initializer=tf.constant_initializer(bias))
    else:
        conv_b = None

    # Quantization Training
    q_conv_w = qu._binarize_tf_weight(conv_w)
    conv_w = conv_w + tf.stop_gradient(q_conv_w - conv_w)

    convs = tf.nn.conv2d(input_, conv_w, strides=[1,c_s,c_s,1], padding='SAME', name='conv')
    if bias is not None:
        convs = tf.nn.bias_add(convs, conv_b, name='preact')
    L = tf.contrib.layers.batch_norm(convs, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L, name='Activation')

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd, name='pool')
        return conv_w, conv_b, L
    return conv_w, conv_b, L

def binary_conv_per_channel(input_, output_channel, bias=None, c_s=1, k_s=3, pool=False, p_k=2, p_s=2, pd='VALID', act=True, name='conv_w'):
    input_shape = input_.get_shape().as_list()
    input_shape = input_shape[3]
    conv_w = tf.get_variable(name=name, shape=[k_s, k_s, input_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())

    # conv_w = tf.Variable(tf.random_uniform(shape=[k_s, k_s, input_shape, output_channel]), name='conv_w')
    if bias is not None:
        conv_b = tf.get_variable(name='conv_b', shape=[output_channel], 
                                initializer=tf.constant_initializer(bias))
    else:
        conv_b = None

    # Quantization Training
    output_list = []
    for idx in range(output_channel):
        channel = conv_w[:,:,:,idx]
        q_conv_channel = qu._binarize_tf_weight(channel)
        channel = channel + tf.stop_gradient(q_conv_channel - channel)
        output_list.append(channel)
    conv_w = tf.stack(output_list, axis=3)
    # conv_w = conv_w + tf.stop_gradient(q_conv_w - conv_w)

    convs = tf.nn.conv2d(input_, conv_w, strides=[1,c_s,c_s,1], padding='SAME', name='conv')
    if bias is not None:
        convs = tf.nn.bias_add(convs, conv_b, name='preact')
    L = tf.contrib.layers.batch_norm(convs, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L, name='Activation')

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd, name='pool')
        return conv_w, conv_b, L
    return conv_w, conv_b, L

def binary_dense(input_, output_channel, dropout, fully=False, hypothesis=False):
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
        fc_w = tf.get_variable(name='fc_w', shape=[flat_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())
    else:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]       
        fc_w = tf.get_variable(name='fc_w', shape=[input_shape, output_channel], 
                                initializer=tf.contrib.layers.xavier_initializer())

    fc_b = tf.get_variable(name='fc_b', shape=[output_channel], initializer=tf.constant_initializer(0.0))
    
    # Quantization Training
    q_fc_w= qu._binarize_tf_weight(fc_w)
    fc_w = fc_w + tf.stop_gradient(q_fc_w - fc_w)

    fc_ = tf.matmul(input_, q_fc_w)
    fc_ = fc_ + fc_b
    if hypothesis:
        return fc_w, fc_b, fc_
    else:
        L1 = tf.contrib.layers.batch_norm(fc_, scale=True, is_training=True)
        L1 = tf.nn.relu(fc_, name='Relu')
        if dropout is not None:
            L1 = tf.nn.dropout(L1, keep_prob=dropout, name='dropout')
        return fc_w, fc_b, L1

def extend_bit_binary_conv(input_, param_lst, bias=True, name='', c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', count=0, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    if bias:
        input_b = param_lst[tmp+1]
    else:
        input_b = None
    
    Q_conv_w, Q_conv_w_s = qu._binarize_alpa_level(input_w)
    # conv_random_matrix = tf.get_variable(name='conv_random'+name, shape=Q_conv_w.get_shape(), initializer=tf.constant_initializer(0.0))
    conv_random_matrix = tf.Variable(tf.random_normal(shape=Q_conv_w.get_shape(), mean=0.0, stddev=0.3), name='conv_random'+name)
    extend_conv_w, extend_conv_w_scale = qu._binary_extend_bit(level=Q_conv_w, scale=Q_conv_w_s, 
                random_matrix=conv_random_matrix, name='new_conv_w', qnum=1, expend_bit=expend_bit)
    L = qu.conv_2d(inputs=input_, level=extend_conv_w, bias=input_b, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w') 

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        return extend_conv_w, extend_conv_w_scale, L

    return extend_conv_w, extend_conv_w_scale, L

def extend_bit_binary_conv_channel(input_, param_lst, bias=True, name='', c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', count=0):
    tmp = count
    input_w = param_lst[tmp]
    if bias:
        input_b = param_lst[tmp+1]
    else:
        input_b = None
    
    input_w_shape = input_w.get_shape().as_list()
    out_channel = input_w_shape[3]
    output_list = []
    for idx in range(out_channel):
        channel = input_w[:,:,:,idx]
        Q_conv_w, Q_conv_w_s = qu._binarize_alpa_level(channel)
        conv_random_matrix = tf.get_variable(name='conv_random_'+name+'_'+str(idx), shape=Q_conv_w.get_shape(), initializer=tf.constant_initializer(0.0))
        extend_conv_w, extend_conv_w_scale = qu._binary_extend_bit(level=Q_conv_w, scale=Q_conv_w_s, 
                    random_matrix=conv_random_matrix, name='new_conv_w', qnum=1)
        channel = tf.scalar_mul(scalar=extend_conv_w_scale, x=extend_conv_w, name='dequant_channel')
        output_list.append(channel)
    input_w = tf.stack(output_list, axis=3)
    L = qu.conv_2d_per_channel(inputs=input_, weight=input_w, bias=input_b, strides=c_s, name='dequant_conv_w') 

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        return extend_conv_w, extend_conv_w_scale, L

    return extend_conv_w, extend_conv_w_scale, L

def extend_bit_binary_dense(input_, param_lst, dropout, fully=False, hypothesis=False, count=10, expend_bit=1):
    tmp = count
    input_w = param_lst[tmp]
    input_b = param_lst[tmp+1]
    
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
    Q_fc_w, Q_fc_w_s = qu._binarize_alpa_level(input_w)
    # fc_random_matrix = tf.get_variable(name='fc_random', shape=Q_fc_w.get_shape(), 
    #                                  initializer=tf.constant_initializer(0.0))
    fc_random_matrix = tf.Variable(tf.random_normal(shape=Q_fc_w.get_shape(), mean=0.0, stddev=0.3), name='fc_random')
    extend_fc_w, extend_fc_w_scale = qu._binary_extend_bit(level=Q_fc_w, scale=Q_fc_w_s, 
        random_matrix=fc_random_matrix, name='new_fc_w', qnum=1, expend_bit=expend_bit)
    if hypothesis:
        L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='', name='dequant_fc_w')
        return extend_fc_w, extend_fc_w_scale, L
    L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='relu', name='dequant_fc_w')
    L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L

def vg_binary_dense(input_, output_channel, dropout=None, fully=False, hypothesis=False):
    fc_w, fc_b, fc_ = binary_dense(input_, output_channel, dropout=dropout, fully=fully, hypothesis=hypothesis)
    return fc_w, fc_b, fc_

def precision_up_conv(input_, level, scale, qnum, bias=True, name='', c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', act=True):
    scale_rate = 1
    expend_bit = 1
    conv_random_matrix = tf.get_variable(name='conv_random_'+name, shape=level.shape, 
                                            initializer=tf.constant_initializer(0.0))
    extend_conv_w, extend_conv_w_scale = qu._quant_extend_bit(quant_w=level, scale=scale, 
                random_matrix=conv_random_matrix, expend_bit=expend_bit, name='new_conv_w_'+name, scale_rate=scale_rate, qnum=qnum)
    L = qu.only_conv_2d(inputs=input_, level=extend_conv_w, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w_'+name)
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L)

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)

    return extend_conv_w, extend_conv_w_scale, L

def precision_up_dense(input_, level, scale, bias, qnum, dropout=None, fully=False, hypothesis=False):
    scale_rate = 1
    expend_bit = 1
    if bias is not None:
        input_b = bias
    else:
        input_b = None
    if fully:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, input_shape])
    fc_random_matrix = tf.get_variable(name='fc_random', shape=level.shape, 
                                        initializer=tf.constant_initializer(0.0))  
    extend_fc_w, extend_fc_w_scale = qu._quant_extend_bit(quant_w=level, scale=scale, 
        random_matrix=fc_random_matrix, expend_bit=expend_bit, name='new_fc_w', scale_rate=scale_rate, qnum=qnum)
    L = qu.only_denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, name='dequant_fc_w')

    if hypothesis:
        return extend_fc_w, extend_fc_w_scale, L
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L

def up_conv(input_, output_channel, qnum, param_lst, bias=None, c_s=1, k_s=3, pool=False, p_k=2, p_s=2, pd='VALID', act=True, name='conv_w', count=0):
    input_shape = input_.get_shape().as_list()
    input_shape = input_shape[3]
    tmp = count
    weight_value = param_lst[tmp]
    conv_w = tf.get_variable(name=name, shape=[k_s, k_s, input_shape, output_channel], 
                                initializer=tf.constant_initializer(weight_value))
    if bias is not None:
        conv_b = tf.get_variable(name='conv_b', shape=[output_channel], 
                                initializer=tf.constant_initializer(bias))
    else:
        conv_b = None

    # Quantization Training
    q_conv_w, _ = qu._quantize_scale_print(conv_w, qnum=qnum)
    conv_w = conv_w + tf.stop_gradient(q_conv_w - conv_w)

    convs = tf.nn.conv2d(input_, conv_w, strides=[1,c_s,c_s,1], padding='SAME', name='conv')
    if bias is not None:
        convs = tf.nn.bias_add(convs, conv_b, name='preact')
    L = tf.contrib.layers.batch_norm(convs, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L, name='Activation')

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd, name='pool')
        return conv_w, conv_b, L
    return conv_w, conv_b, L

def up_dense(input_, output_channel, qnum, param_lst, dropout=None, fully=False, hypothesis=False, count=0):
    tmp = count
    weight_value = param_lst[tmp]
    bias_value = param_lst[tmp+1]
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])
        fc_w = tf.get_variable(name='fc_w', shape=[flat_shape, output_channel], 
                                initializer=tf.constant_initializer(weight_value))
    else:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1] 
        fc_w = tf.get_variable(name='fc_w', shape=[input_shape, output_channel], 
                                initializer=tf.constant_initializer(weight_value))

    fc_b = tf.get_variable(name='fc_b', shape=[output_channel], initializer=tf.constant_initializer(bias_value))
    
    # Quantization Training
    q_fc_w, _ = qu._quantize_scale_print(fc_w, qnum=qnum)
    fc_w = fc_w + tf.stop_gradient(q_fc_w - fc_w)

    fc_ = tf.matmul(input_, fc_w)
    fc_ = fc_ + fc_b
    if hypothesis:
        return fc_w, fc_b, fc_
    else:
        L1 = tf.contrib.layers.batch_norm(fc_, scale=True, is_training=True)
        L1 = tf.nn.relu(L1, name='Relu')
        if dropout is not None:
            L1 = tf.nn.dropout(L1, keep_prob=dropout, name='dropout')
        return fc_w, fc_b, L1

def vgg_e_up_conv(input_, param_list, qnum, name, c_s=1, act=True, pool=False, p_k=2, p_s=2, pd='VALID', count=0, dropout=None):
    scale_rate = 1
    expend_bit = 1
    # input_w = param_list[count]
    # Q_conv_w, Q_conv_w_s = qu.class_quantize(input_w, quantize_num=qnum)
    Q_conv_w = param_list[count]
    Q_conv_w = tf.Variable(initial_value=Q_conv_w, trainable=False)
    Q_conv_w_s = param_list[count+1]
    Q_conv_w_s = tf.Variable(initial_value=Q_conv_w_s, trainable=False)
    # conv1_random_matrix = tf.Variable(tf.random_uniform(shape=quant_conv_w1.get_shape(), minval=0, maxval=random_uniform_maxval), name='conv1_random')
    conv_random_matrix = tf.get_variable(name='conv_random_'+name, shape=Q_conv_w.shape, 
                                            initializer=tf.constant_initializer(0.0))
    extend_conv_w, extend_conv_w_scale = qu._quant_extend_bit(quant_w=Q_conv_w, scale=Q_conv_w_s, 
                random_matrix=conv_random_matrix, expend_bit=expend_bit, name='new_conv_w_'+name, scale_rate=scale_rate, qnum=qnum)
    L = qu.only_conv_2d(inputs=input_, level=extend_conv_w, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w_'+name)
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    if act:
        L = tf.nn.relu(L)

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        if dropout is not None:
            L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_conv_w, extend_conv_w_scale, L

def vgg_e_up_dense(input_, param_lst, qnum, dropout=None, count=13, fully=False, hypothesis=False):
    scale_rate = 1
    expend_bit = 1

    tmp = count
    # input_w = param_lst[tmp]
    # input_b = param_lst[tmp+1]
    quant_fc_w = param_lst[tmp]
    quant_fc_w = tf.Variable(initial_value=quant_fc_w, trainable=False)
    quant_fc_w_scale = param_lst[tmp+1]
    quant_fc_w_scale = tf.Variable(initial_value=quant_fc_w_scale, trainable=False)
    input_b = param_lst[tmp+2]
    input_b = tf.Variable(initial_value=input_b, trainable=False)

    if fully:
        input_shape = input_.get_shape().as_list()
        input_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, input_shape])
    # quant_fc_w, quant_fc_w_scale = qu.class_quantize(input_w, quantize_num=qnum)
    # fc6_random_matrix = tf.Variable(tf.random_uniform(shape=quant_fc_w6.get_shape(), minval=0, maxval=random_uniform_maxval), name='fc6_random')
    fc_random_matrix = tf.get_variable(name='fc_random', shape=quant_fc_w.get_shape(), 
                                        initializer=tf.constant_initializer(0.0))  
    extend_fc_w, extend_fc_w_scale = qu._quant_extend_bit(quant_w=quant_fc_w, scale=quant_fc_w_scale, 
        random_matrix=fc_random_matrix, expend_bit=expend_bit, name='new_fc_w', scale_rate=scale_rate, qnum=qnum)
    L = qu.only_denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, name='dequant_fc_w')

    if hypothesis:
        return extend_fc_w, extend_fc_w_scale, L
    L = tf.contrib.layers.batch_norm(L, scale=True, is_training=True)
    L = tf.nn.relu(L)
    if dropout is not None:
        L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L

def alex_e_up_conv(input_, param_list, qnum, bias=True, c_s=1, pool=False, p_k=2, p_s=2, pd='VALID', count=0, dropout=None):

    Q_conv_w = param_list[count]
    Q_conv_w = tf.Variable(initial_value=Q_conv_w, trainable=False)
    input_b = param_list[count+1]
    input_b = tf.Variable(initial_value=input_b, trainable=False)
    Q_conv_w_s = param_list[count+2]
    Q_conv_w_s = tf.Variable(initial_value=Q_conv_w_s, trainable=False)
    # conv1_random_matrix = tf.Variable(tf.random_uniform(shape=quant_conv_w1.get_shape(), minval=0, maxval=random_uniform_maxval), name='conv1_random')
    conv_random_matrix = tf.get_variable(name='conv_random', shape=Q_conv_w.get_shape(), initializer=tf.constant_initializer(0.0))
    extend_conv_w, extend_conv_w_scale = qu._quant_extend_bit(quant_w=Q_conv_w, scale=Q_conv_w_s, 
                random_matrix=conv_random_matrix, name='new_conv_w', qnum=qnum)
    L = qu.conv_2d(inputs=input_, level=extend_conv_w, bias=input_b, scale=extend_conv_w_scale, strides=c_s, name='dequant_conv_w') 

    if pool:
        L = tf.nn.max_pool(L, ksize=[1, p_k, p_k, 1], strides=[1, p_s, p_s, 1], padding=pd)
        if dropout is not None:
            L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_conv_w, extend_conv_w_scale, L

def alex_e_up_dense(input_, param_list, qnum, dropout, fully=False, hypothesis=False, count=10):

    Q_fc_w = param_list[count]
    Q_fc_w = tf.Variable(initial_value=Q_fc_w, trainable=False)
    input_b = param_list[count+1]
    input_b = tf.Variable(initial_value=input_b, trainable=False)
    Q_fc_w_s = param_list[count+2]
    Q_fc_w_s = tf.Variable(initial_value=Q_fc_w_s, trainable=False)
    
    if fully:
        input_shape = input_.get_shape().as_list()
        flat_shape = input_shape[1]*input_shape[2]*input_shape[3]
        input_ = tf.reshape(input_, shape=[-1, flat_shape])

    fc_random_matrix = tf.get_variable(name='fc_random', shape=Q_fc_w.get_shape(), 
                                        initializer=tf.constant_initializer(0.0))
    extend_fc_w, extend_fc_w_scale = qu._quant_extend_bit(quant_w=Q_fc_w, scale=Q_fc_w_s, 
        random_matrix=fc_random_matrix, name='new_fc_w', qnum=qnum)
    if hypothesis:
        L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='', name='dequant_fc_w')
        return extend_fc_w, extend_fc_w_scale, L
    L = qu.denselayer(x=input_, l=extend_fc_w, b=input_b, scale=extend_fc_w_scale, activation='relu', name='dequant_fc_w')
    L = tf.nn.dropout(L, keep_prob=dropout, name='dropout')

    return extend_fc_w, extend_fc_w_scale, L
