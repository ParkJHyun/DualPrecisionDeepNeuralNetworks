import warnings
warnings.filterwarnings('ignore')
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from alexnet_alter_graph import AlexNet_cifar100, Train_Alexnet
from dataset_helper import read_cifar_10
import train_utils as tu

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

INPUT_WIDTH = 32
INPUT_HEIGHT = 32
BATCH_SIZE = 128
CHECKPOINT = "./Train/train_2bit_4bit/scale_0.3"

print('Reading CIFAR-10...')
X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

X = tf.placeholder(tf.float32, [None, 32, 32, 3]) 
Y = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder("float")

fix_model = AlexNet_cifar100(X, qnum=2, dropout_keep_prob=dropout_rate)
param_list = fix_model.parameter_list

tr_model = Train_Alexnet(X, param_list, dropout_keep_prob=dropout_rate)
var_all = tf.trainable_variables(scope=None)

hypothesis = tr_model.hypothesis

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) , name='accuracy')

with tf.device('/GPU:0'): 
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            sess.run(tf.global_variables_initializer())
                        
            loader = tf.train.Saver(var_all)
            loader.restore(sess, tf.train.latest_checkpoint(CHECKPOINT))

        final_train_accuracy = tu.evaluate(sess, accuracy, X, Y, dropout_rate, X_train, Y_train, BATCH_SIZE)
        final_test_accuracy = tu.evaluate(sess, accuracy, X, Y, dropout_rate, X_test, Y_test, BATCH_SIZE)

        print('Train Accuracy = {:.3f}'.format(final_train_accuracy))
        print('Test Accuracy = {:.3f}'.format(final_test_accuracy))
        print("")