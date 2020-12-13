import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def class_quantize(weight, quantize_num):
    
    '''
        Linear symmetric quantization
        
        quantize_num-bit quantization
    '''

    if quantize_num > 2:
        # low_weights = tf.reduce_min(weight)
        # high_weights = tf.reduce_max(weight)
        # vmax = high_weights - low_weights
        # abs_weights = tf.abs(weight)
        vmax = tf.reduce_max(weight)
        quant_max = (2**(quantize_num-1)) -1
        scale = tf.divide(vmax, quant_max)
    else:
        vmax = tf.abs(weight)
        vmax = tf.reduce_max(vmax)
        # Assume vlaues have sign
        quant_max = 1
        scale = tf.divide(vmax, 2)

    scale_f = tf.divide(weight, tf.cast(scale, tf.float32))

    r_value = tf.round(scale_f)
    quantize_level = tf.clip_by_value(r_value, -1*(quant_max + 1), quant_max)

    return quantize_level, scale

def conv_2d(inputs, level, bias=None, scale=0., strides=1, padding='SAME', name='dequant_conv'):
    # weight = tf.multiply(level, scale, name=name)
    weight = tf.math.scalar_mul(scalar=scale, x=level, name=name)

    x = tf.nn.conv2d(inputs, weight, strides=[1, strides, strides, 1], padding=padding)

    if bias:
        x = tf.nn.bias_add(x, bias)

    x = tf.contrib.layers.batch_norm(x, scale=True, is_training=True)

    x = tf.nn.relu(x)

    # with tf.device('/cpu:0'):
    #     tf.summary.histogram('dequant_conv_w1', weight)

    return x

def conv_2d_per_channel(inputs, weight, bias=None, strides=1, padding='SAME', name='dequant_conv'):
    # weight = tf.multiply(level, scale, name=name)

    x = tf.nn.conv2d(inputs, weight, strides=[1, strides, strides, 1], padding=padding)

    if bias:
        x = tf.nn.bias_add(x, bias)

    x = tf.contrib.layers.batch_norm(x, scale=True, is_training=True)

    x = tf.nn.relu(x)

    # with tf.device('/cpu:0'):
    #     tf.summary.histogram('dequant_conv_w1', weight)

    return x

def denselayer(x, l, b, scale=0., activation='', name='dequant_fc'):
    # w = tf.multiply(l, scale, name=name)
    w = tf.math.scalar_mul(scalar=scale, x=l, name=name)

    x = tf.matmul(x, w)   
    x = tf.nn.bias_add(x, b)

    x = tf.contrib.layers.batch_norm(x, scale=True, is_training=True)

    if activation == "relu":
        x = tf.nn.relu(x)
    elif activation == 'softmax':
        x = tf.nn.softmax(x)
    else:
        return x
    return x

def only_conv_2d(inputs, level, bias=None, scale=0., strides=1, padding='SAME', name='dequant_conv'):
    #weight = tf.multiply(level, scale, name=name)
    weight = tf.math.scalar_mul(scalar=scale, x=level, name=name)
        
    x = tf.nn.conv2d(inputs, weight, strides=[1, strides, strides, 1], padding=padding)

    if bias:
        x = tf.nn.bias_add(x, bias)

    # with tf.device('/cpu:0'):
    #     tf.summary.histogram('dequant_conv_w1', weight)

    return x

def only_denselayer(x, l, b=None, scale=0., name='dequant_fc'):
    #w = tf.multiply(l, scale, name=name)
    w = tf.math.scalar_mul(scalar=scale, x=l, name=name)

    x = tf.matmul(x, w)
    if b is not None:   
        x = tf.nn.bias_add(x, b)
    
    return x

def conv_2d_wgt(inputs, level, bias=None, scale=0., strides=1, padding='SAME', name='dequant_conv'):
    # weight = tf.multiply(level, scale, name=name)
    weight = tf.math.scalar_mul(scalar=scale, x=level, name=name)

    x = tf.nn.conv2d(inputs, weight, strides=[1, strides, strides, 1], padding=padding)

    if bias:
        x = tf.nn.bias_add(x, bias)

    x = tf.nn.relu(x)

    # with tf.device('/cpu:0'):
    #     tf.summary.histogram('dequant_conv_w1', weight)

    return x, weight

def denselayer_wgt(x, l, b, scale=0., activation='', name='dequant_fc'):
    # w = tf.multiply(l, scale, name=name)
    w = tf.math.scalar_mul(scalar=scale, x=l, name=name)

    x = tf.matmul(x, w)   
    x = tf.nn.bias_add(x, b)

    if activation == "relu":
        x = tf.nn.relu(x)
    elif activation == 'softmax':
        x = tf.nn.softmax(x)
    else:
        return x
    return x, w  


def __sat(x, numbits):
    r = tf.round(x)
    return tf.clip_by_value(r, -1*2**(numbits-1), 2**(numbits-1)-1)

def __quantize(weights, qnum):
    abs_value = tf.abs(weights)
    vmax = tf.reduce_max(abs_value)

    quant_max = (2**(qnum-1))-1

    s = tf.divide(vmax, quant_max)
    x = tf.divide(weights, tf.cast(s, tf.float32))
    x = __sat(x, numbits=qnum)

    ##Dequantize
    x = tf.multiply(x, s)

    return x


def _quantize_scale_print(weights, qnum):
    if qnum > 2:
        # low_weights = tf.reduce_min(weights)
        # abs_weights = tf.abs(weights)
        vmax = tf.reduce_max(weights)
        quant_max = (2**(qnum-1)) -1
        s = tf.divide(vmax, quant_max)
        # vmax = high_weights - low_weights
    else:
        vmax= tf.abs(weights)
        vmax = tf.reduce_max(vmax)
        quant_max = 2
        s = tf.divide(vmax, quant_max)

    x = tf.divide(weights, tf.cast(s, tf.float32))
    x = __sat(x, numbits=qnum)

    ##Dequantize
    x = tf.multiply(x, s)

    return x, s

def _quant_extend_bit(quant_w, scale, random_matrix, name, qnum, expend_bit=1, scale_rate=1):
    '''
    This function expends quantization bits(ex. 2bit -> 3bit).

    ex) 2bit, level has -1, 0, 1 -> 3bit, level has -2

    Args:

    `quant_w`, `scale` = quantized weight index(level) & scale using class_quantize func.

    `random_matrix` = For a expending 1bit, 1bit is trainable. This matrix will be add to modified weight level.

    funcs:
        
    `double_scale` -> scale / 2
        
    `double_weight_level` -> quant_w(level index) * 2

    `differ_round` make random_matrix enable to be rounded with gradient using tf.stop_gradient.

    `results_weight_level` -> double_weight_level + differ_round 
    '''
    # scale_rate = 0.25
    # Double Scale = 2 * original_rate
    original_rate = pow(2, qnum) - 1
    # original_rate = 2 * original_rate
    additional_rate = pow(2, qnum+expend_bit) - 1
    additional_rate = 4 * additional_rate
    scale = original_rate * scale
    expend_scale = additional_rate

    #expend_scale = pow(2, expend_bit) * scale_rate
    expend_level = pow(2, expend_bit)
        
    double_scale = tf.divide(scale, expend_scale)
    double_weight_level = tf.multiply(quant_w, expend_level)

    # Round = x + Threshold
    # Threshold = 0.5 - t 

    # threshold = tf.constant(0.8, shape=random_matrix.get_shape())

    # random_max = tf.reduce_max(random_matrix)
    # random_min = tf.reduce_min(random_matrix)
    # random_range = random_max - random_min

    scale_range = expend_level
    # new_random_matrix = random_matrix * (scale_range/random_range)
    
    stop_differ_round = tf.round(random_matrix)
    stop_differ_round = tf.clip_by_value(stop_differ_round, 0, scale_range-1)
    differ_round = random_matrix + tf.stop_gradient(stop_differ_round - random_matrix)

    results_weight_level = tf.add(double_weight_level, differ_round)
    return results_weight_level, double_scale


def log(x, base=2):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def _log_round(x):
    abs_x = tf.abs(x)
    x = log(abs_x)
    
    halfprecision = (2**(tf.ceil(x)) - 2**(tf.floor(x)))/2
    fractional = 2**(x) - 2**(tf.floor(x))
    condition = tf.greater(fractional, halfprecision)
    return tf.where(condition, tf.ceil(x), tf.floor(x))

def _log_clip(x, bitwidth):
    vmax = tf.reduce_max(x)
    vmin = tf.reduce_min(x)
    FSR = log((vmax - vmin))

    rmin = FSR - 2**(bitwidth)
    rmax = FSR
    rr_max = tf.fill(x.get_shape(), rmax-1)

    mask1 = tf.greater(x, rmin)
    mask1 = tf.multiply(tf.cast(mask1, tf.float32), x)
    mask2 = tf.greater(mask1, rmax)
    
    cond = tf.where(mask2, rr_max, mask1)

    return cond

def _log_quant(weights, qnum):
    non_zero_mask = tf.not_equal(weights, 0)

    x = _log_round(weights)
    x = _log_clip(x, qnum)

    x = tf.pow(2.0, x)

    result = tf.cast(non_zero_mask, tf.float32) * x

    return result


def _log_round2(lookup, x, fc=False):
  distance = tf.reshape(lookup, (-1, 1, 1, 1, 1))
  distance = tf.abs(x - distance)
  distance = tf.transpose(distance, (1, 2, 3, 4, 0))
  idx = tf.argmin(distance, axis=-1)
  idx = tf.reshape(idx, x.get_shape())
  return idx

def _log_quant2(weights, qnum, lookup, fc=False):
    sign_mask = tf.sign(weights)
    abs_weigths = tf.abs(weights)
    log_weights = log(abs_weigths)
    idx = _log_round2(lookup, log_weights, fc)
    result = tf.multiply(tf.cast(idx, tf.float32), lookup[1])
    result = tf.pow(2.0, result)
    result = result * sign_mask

    return result

def extend_log_quantize(weight, lookup, random_matrix, expend_bit=1, quantize_num=2, dense=False):
    
    '''
        Log quantization
        
        Extend bit quantize
    '''
    scale = lookup[1]
    sign_mask = tf.sign(weight)
    abs_weigths = tf.abs(weight)
    log_weights = log(abs_weigths)
    idx = _log_round2(lookup, log_weights, fc=dense)

    expend_level = pow(2, expend_bit)
    double_idx = idx * expend_level
    expend_scale = scale / 2

    stop_differ_round = tf.round(random_matrix)
    stop_differ_round = tf.clip_by_value(stop_differ_round, 0, expend_level-1)
    differ_round = random_matrix + tf.stop_gradient(stop_differ_round - random_matrix)

    results_weight_level = tf.add(tf.cast(double_idx, tf.float32), differ_round)
    result_weight = tf.multiply(tf.cast(results_weight_level, tf.float32), expend_scale)
    results = tf.pow(2.0, result_weight)
    results = results*sign_mask

    return results_weight_level, expend_scale, results


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def binarize_weights(x, name=None):
    """Creates the binarize_weights Op with f as forward pass
    and df as the gradient for the backward pass
    Args:
        x: The input Tensor
        name: the name for the Op
    
    Returns:
        The output tensor
    """

    def _binarize_each_channels(weights):
      if weights.ndim > 2:
          index = np.size(weights, 3)
          for i in range(index):
            channel = weights[:,:,:,i]
            channel = _np_binarize_(channel)
            weights[:,:,:,i] = channel
      else:
          index = np.size(weights, 1)
          for i in range(index):
            channel = weights[:,i]
            channel = _np_binarize_(channel)
            weights[:,i] = channel
      return weights

    def _np_binarize_(weight):
      abs_value = np.abs(weight)
      vmax = np.max(abs_value)
      r_weight = np.sign(weight)
      cond = np.equal(r_weight, 0.)
      result = np.clip(r_weight, -1, 1)
      result = np.add(result, cond.astype(np.float32))
      result = result * vmax
      
      return result

    def df(op, grad):
        x = op.inputs[0]
        if x.get_shape().ndims > 2:
          print("4")
          n = tf.reduce_prod(tf.shape(x[:,:,:,0])[:3])
          alpha = tf.div(tf.reduce_sum(tf.abs(x), [0, 1, 2]), tf.cast(n, tf.float32))
          ds = tf.multiply(x, tf.cast(tf.less_equal(tf.abs(x), 1), tf.float32))
        else:
          print("2")
          n = tf.reduce_prod(tf.shape(x[:,0])[:3])
          alpha = tf.div(tf.reduce_sum(tf.abs(x), [0]), tf.cast(n, tf.float32))
          ds = tf.multiply(x, tf.cast(tf.less_equal(tf.abs(x), 1), tf.float32))
        return tf.multiply(grad, tf.add(tf.cast(1/n, tf.float32), tf.multiply(alpha, ds)))
        
    with ops.name_scope(name, 'BinarizeWeights', [x]) as name:
        fx = py_func(_np_binarize_, [x], [tf.float32], name=name, grad=df)
        return fx[0]

def _binarize_alpa_level(weight):
    abs_value = tf.abs(weight)
    vmax = tf.reduce_max(abs_value)
    sign_weight = tf.sign(weight)
    cond = tf.equal(sign_weight, 0.)
    level = tf.add(sign_weight, tf.cast(cond, tf.float32))

    return level, vmax

def _binarize_tf_weight(weight):
    abs_value = tf.abs(weight)
    vmax = tf.reduce_max(abs_value)
    sign_weight = tf.sign(weight)
    cond = tf.equal(sign_weight, 0.)
    level = tf.add(sign_weight, tf.cast(cond, tf.float32))
    weight = tf.multiply(level, vmax)

    return weight

def _binary_extend_bit(level, scale, random_matrix, name, expend_bit=1, scale_rate=1, qnum=2):
    '''
    This function expends quantization bits(ex. 2bit -> 3bit).

    ex) 2bit, level has -1, 0, 1 -> 3bit, level has -2

    Args:

    `quant_w`, `scale` = quantized weight index(level) & scale using class_quantize func.

    `random_matrix` = For a expending 1bit, 1bit is trainable. This matrix will be add to modified weight level.

    funcs:
        
    `double_scale` -> scale / 2
        
    `double_weight_level` -> quant_w(level index) * 2

    `differ_round` make random_matrix enable to be rounded with gradient using tf.stop_gradient.

    `results_weight_level` -> double_weight_level + differ_round 
    '''
    
    expend_level = pow(2, expend_bit-1)
    level = level * expend_level
    
    random_max = tf.reduce_max(random_matrix)
    random_min = tf.reduce_min(random_matrix)
    random_range = random_max - random_min
    
    new_random_matrix = random_matrix * (2*expend_level / random_range)

    stop_differ_round = tf.round(new_random_matrix)
    # stop_differ_round = tf.sign(random_matrix)
    stop_differ_round = tf.clip_by_value(stop_differ_round, -1 * pow(2, expend_bit-1), pow(2, expend_bit-1)-1)
    # stop_differ_round = tf.multiply(stop_differ_round, -1)
    differ_round = random_matrix + tf.stop_gradient(stop_differ_round - random_matrix)
    
    additional_rate = pow(2, expend_bit+1) - 1
    # expend_scale = 4 * additional_rate
    expend_scale = additional_rate
    double_scale = tf.divide(scale, expend_scale)

    results_weight_level = tf.add(level, differ_round)
    # double_cond1 = tf.add(double_cond1, differ_round)
    # results_weight_level = tf.add(cond2, double_cond1)

    return results_weight_level, double_scale
 
