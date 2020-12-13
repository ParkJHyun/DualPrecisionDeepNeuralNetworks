import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import sys
from absl import app, flags
# pylint: disable=unbalanced-tuple-unpacking

import tensorflow as tf
from dataset_helper import get_dataset, check_available_gpus
import original_utils as utils
from figure import plot_acc
from log_helper import LogManager

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu_num = check_available_gpus()

FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu_nums', gpu_num, 'available gpus')
flags.DEFINE_integer('batch_size', 125, 'batch size for train.')
flags.DEFINE_integer('num_bit', 4, 'bit for quantize')
flags.DEFINE_integer('up_bit', 0, 'bit for quantize')
flags.DEFINE_integer('epochs', 200, 'batch size for train.')
flags.DEFINE_enum('dataset', 'CIFAR10', ['CIFAR10', 'CIFAR100', 'imagenet2012',
                                    'mnist', 'fashion_mnist', 'imagenet_resized/64x64', 'imagenet_resized/32x32'],
                                    'Choose dataset.')
flags.DEFINE_enum('model', 'resnet20', ['alexnet', 'vgg13','vgg16', 'vgg19', 'resnet20', 'resnet50', 'resnet110'],
                                    'Choose mosdel.')
flags.DEFINE_enum('q_mode', 'linear', ['linear', 'log', 'binary', 'upgrade'],
                                    'Choose quantization model.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'dropout rate.')

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('q_mode')

def main(argv):
    del argv
    save, logdir, figname, logHandler = utils.configuration(FLAGS)

    
    train_ds, test_ds, placeholder = get_dataset(FLAGS)
    loss, fix_accuracy, var_m1 = utils.load_model(FLAGS, placeholder)

    train_iterator = None
    test_iterator = None

    fix_opt = utils.make_optimizer(placeholder, loss, var_m1)

    _, save_file = save

    epoch_list, original = [], []

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir)
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_m1)

        print('Learning started. It takes sometimes...')
        print()
        for i in range(1, FLAGS.epochs+1):
            logHandler.print_epoch()
            utils.fit_model(sess, fix_opt, placeholder, train_iterator,
                            train_ds, i, FLAGS, logHandler, merged_summary, writer)

            utils.train_validate(sess, fix_accuracy, train_iterator, 
                                                placeholder, train_ds, FLAGS, logHandler)
            origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                placeholder, test_ds, FLAGS, logHandler)
                
            saver.save(sess, save_file)              
                    
            epoch_list.append(i)
            original.append(origin_test_accuracy)

        logHandler._print("Learning Finished!!")

        logHandler._print('Original Accuracy: ')
        origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                placeholder, test_ds, FLAGS, logHandler)

        plot_acc(epoch_list, original, None, figname)
        saver.save(sess, save_file)
        
    logHandler._print('Training done successfully')

if __name__ == '__main__':
    app.run(main)
