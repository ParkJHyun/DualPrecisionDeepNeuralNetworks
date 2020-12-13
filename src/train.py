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
import train_utils as utils
from figure import plot_acc
from log_helper import LogManager

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu_num = check_available_gpus()

FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu_nums', gpu_num, 'available gpus')
flags.DEFINE_integer('batch_size', 125, 'batch size for train.')
flags.DEFINE_integer('num_bit', 2, 'bit for quantize')
flags.DEFINE_integer('up_bit', 1, 'extend bit')
flags.DEFINE_integer('epochs', 100, 'batch size for train.')
flags.DEFINE_integer('stop_point', 50, 'num of epochs to run.')
flags.DEFINE_enum('dataset', 'CIFAR10', ['CIFAR10', 'CIFAR100', 'imagenet2012',
                                    'mnist', 'fashion_mnist', 'imagenet_resized/64x64', 'imagenet_resized/32x32'],
                                    'Choose dataset.')
flags.DEFINE_enum('model', 'vgg16', ['alexnet', 'vgg13','vgg16', 'vgg19', 'resnet50', 'resnet110'],
                                    'Choose model.')
flags.DEFINE_enum('q_mode', 'linear', ['linear', 'log', 'binary', 'upgrade'],
                                    'Choose quantization model.')
flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')
flags.DEFINE_float('add_lr', 0.00003, 'Initial learning rate.')
flags.DEFINE_float('stop_lr', 0.0007, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.8, 'dropout rate.')
flags.DEFINE_integer('iteration', 2, 'iteration')
flags.DEFINE_float('factor', 100.0, 'normalize proposed models hypothesis.')


flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('q_mode')

def main(argv):
    del argv
    save, logdir, figname, logHandler = utils.configuration(FLAGS)
    
    train_ds, test_ds, placeholder = get_dataset(FLAGS)
    loss, correct_prediction, var_list = utils.load_model(FLAGS, placeholder)

    train_iterator = None
    test_iterator = None

    fix_opt, add_opt, stop_opt = utils.make_optimizer(placeholder, loss, var_list)
    fix_accuracy, add_accuracy = correct_prediction

    save_dir, save_file = save
    var_all, var_m1, _ = var_list

    epoch_list, original, proposed = [], [], []

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir)
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_all)

        print('Learning started. It takes sometimes...')
        print()
        for i in range(1, FLAGS.epochs+1):
            logHandler.print_epoch()
            if i == (FLAGS.stop_point + 1):
                logHandler._print('Proposed training...')
                loader = tf.train.Saver(var_m1)
                loader.restore(sess, tf.train.latest_checkpoint(save_dir))

            if i <= FLAGS.stop_point:
                if i % FLAGS.iteration == 0:
                    loader = tf.train.Saver(var_all)
                    loader.restore(sess, tf.train.latest_checkpoint(save_dir))

                    utils.fit_model(sess, add_opt, placeholder, train_iterator,
                            train_ds, i, FLAGS, logHandler, merged_summary, writer)

                    origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                        placeholder, test_ds, FLAGS, logHandler)
                    proposed_test_accuracy = utils.test_validate(sess, add_accuracy, test_iterator, 
                                                        placeholder, test_ds, FLAGS, logHandler)

                else:
                    utils.fit_model(sess, fix_opt, placeholder, train_iterator,
                            train_ds, i, FLAGS, logHandler, merged_summary, writer)

                    utils.train_validate(sess, fix_accuracy, train_iterator, 
                                            placeholder, train_ds, FLAGS, logHandler)
                    origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                        placeholder, test_ds, FLAGS, logHandler)
                        
                    proposed_test_accuracy = utils.test_validate(sess, add_accuracy, test_iterator, 
                                                        placeholder, test_ds, FLAGS, logHandler)
                saver.save(sess, save_file)              
            else:
                # loader = tf.train.Saver(var_m1)
                # loader.restore(sess, tf.train.latest_checkpoint(save_dir))
                utils.fit_model(sess, stop_opt, placeholder, train_iterator, 
                        train_ds, i, FLAGS, logHandler, merged_summary, writer)

                if train_iterator is not None:
                    sess.run(train_iterator.initializer)
                utils.train_validate(sess, add_accuracy, train_iterator, 
                                        placeholder, train_ds, FLAGS, logHandler)
                proposed_test_accuracy = utils.test_validate(sess, add_accuracy, test_iterator, 
                                                        placeholder, test_ds, FLAGS, logHandler)

                origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                        placeholder, test_ds, FLAGS, logHandler)
                    
            epoch_list.append(i)
            proposed.append(proposed_test_accuracy)
            original.append(origin_test_accuracy)

        # Add_final_train_accuracy = tu.train_validate(sess, add_accuracy, train_iterator, 
        #                                         X, Y, dropout_rate, train_ds, FLAGS)
        logHandler._print('Original Accuracy: ')
        origin_test_accuracy = utils.test_validate(sess, fix_accuracy, test_iterator,
                                                placeholder, test_ds, FLAGS, logHandler)
        
        logHandler._print('Proposed Accuracy: ')
        utils.test_validate(sess, add_accuracy, test_iterator, 
                                placeholder, test_ds, FLAGS, logHandler)

        plot_acc(epoch_list, original, proposed, figname)
        saver.save(sess, save_file)
        logHandler._print('Training done successfully')

if __name__ == '__main__':
    app.run(main)
