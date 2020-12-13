import sys
import numpy as np
import tensorflow as tf
import os
from log_helper import LogManager
from graph import *
from dataset_helper import data_augmentation, color_preprocessing, color_normalizing

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    # idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def only_onemodel_train(sess, optimizer, Place_X, Place_Y, dropout, X_data, Y_data, step, lr, batch_size, summary_opreation=None, writer=None):
    # From article: We trained our models using stochastic gradient descent with a batch size of 128 examples.
    X = Place_X
    Y = Place_Y
    num_examples = len(X_data)
    merged_summary = summary_opreation
    index = 0
    origin_learning_rate = lr
    starter_learning_rate = 0.009
    dr = 0.5
    if step >= 2:
        starter_learning_rate = starter_learning_rate / 20
        # dr = 0.45
        # if step >= 30:
        #     starter_learning_rate = starter_learning_rate / 2
        #     if step >= 35:
        #         starter_learning_rate = starter_learning_rate / 4

    for offset in range(0, num_examples, batch_size):
        sys.stdout.write("\r %d / %d" % (int(offset), num_examples))
        sys.stdout.flush()

        end = offset + batch_size
        batch_x, batch_y = X_data[offset:end], Y_data[offset:end] 
        batch_x, batch_y = next_batch(batch_size, batch_x, batch_y)

        summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: batch_x, Y: batch_y, dropout:dr, origin_learning_rate:starter_learning_rate})
        writer.add_summary(summary, step * (num_examples // batch_size + 1) + index)
        index += 1

def configuration(FLAGS):
    save_dir = '/workspace/Model/'+FLAGS.model+'/'+FLAGS.dataset+'/original/'+FLAGS.q_mode+'/'+str(FLAGS.num_bit)+'bit/'
    save_file = save_dir+FLAGS.model+'_ext.ckpt'
    logdir = '/workspace/logs/'+FLAGS.q_mode+'/'+FLAGS.dataset+'/original/'+FLAGS.model+'/'+str(FLAGS.num_bit)+'bit'
    Figname = "/workspace/figure/"+FLAGS.dataset+"_"+FLAGS.q_mode+"_"+FLAGS.model+"_"+str(FLAGS.num_bit)+"bit.png"
    log_save = '/workspace/history/'+FLAGS.dataset+'/'+FLAGS.q_mode+'/'+FLAGS.model+'/'+str(FLAGS.num_bit)+'bit.txt'

    if 'imagenet' in FLAGS.dataset:
        total_num_train_data = 1281167
    elif FLAGS.dataset == 'fashion_mnist' or FLAGS.dataset == 'mnist':
        total_num_train_data = 60000
    elif 'CIFAR100' == FLAGS.dataset:
        total_num_train_data = 500000
    else:
        total_num_train_data = 50000

    total_num_batch = total_num_train_data // FLAGS.batch_size
    log_manager = LogManager(FLAGS.epochs, total_num_batch=total_num_batch, log_file_path=log_save)
    log_manager.print_configuration(FLAGS, ensemble=False)

    if not(os.path.isdir(save_dir)):
        os.makedirs(os.path.join(save_dir))
    return (save_dir, save_file), logdir, Figname, log_manager

def load_model(FLAGS, Placeholder):

    '''
        Load model function
        
        1. VGG11, VGG13, VGG16, VGG19

        2. AlexNet

        3. RESNET50, 152(not ready)
    '''

    X, Y, dropout, _ = Placeholder

    fix_losses = []
    fix_correct = []

    X_A = tf.split(X, FLAGS.gpu_nums)
    Y_A = tf.split(Y, FLAGS.gpu_nums)

    if FLAGS.dataset == 'CIFAR10' or FLAGS.dataset == 'MNIST':
        num_class = 10
    elif FLAGS.dataset == 'CIFAR100':
        num_class = 100
    elif 'imagenet' in FLAGS.dataset:
        num_class = 1000
    
    for gpu_id in range(FLAGS.gpu_nums):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    BLOCK = None
                    if 'vgg' in FLAGS.model:
                        if 'vgg16' == FLAGS.model:
                            BLOCK = [2, 2, 3, 3, 3]
                        elif 'vgg19' == FLAGS.model:
                            BLOCK = [2, 2, 4, 4, 4]
                        elif 'vgg13' == FLAGS.model:
                            BLOCK = [2, 2, 2, 2, 2]
                        else:
                            BLOCK = [1, 1, 2, 2, 2]
                        
                        if 'log' == FLAGS.q_mode:
                            Model = VGG_log(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)
                        elif 'linear' == FLAGS.q_mode:
                            Model = VGG_linear(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)
                        elif 'binary' == FLAGS.q_mode:
                            Model = VGG_binary(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)

                    elif 'resnet' in FLAGS.model:
                        if 'resnet50' == FLAGS.model:
                            BLOCK = [3, 4, 6, 3]
                        elif 'resnet110' == FLAGS.model:
                            BLOCK = [3, 4, 23, 3]
                        elif 'resnet152' == FLAGS.model:
                            BLOCK = [3, 8, 36, 3]
                        else:
                            BLOCK = None
                        Model = ResNet20_linear(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK,
                                        num_class=num_class, dropout=dropout, reuse=(gpu_id > 0))
                        # fix_model = ResNet(X_A[gpu_id], qnum=FLAGS.num_bit, block=BLOCK,
                        #                     num_class=num_class, reuse=(gpu_id > 0))

                    elif FLAGS.model == 'alexnet':
                        if 'log' == FLAGS.q_mode:
                            Model = AlexNet_log(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)
                        elif 'linear' == FLAGS.q_mode:
                            Model = AlexNet_linear(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)
                        elif 'binary' == FLAGS.q_mode:
                            Model = AlexNet_binary(X_img=X_A[gpu_id], FLAGS=FLAGS, block=BLOCK, dropout=dropout, num_class=num_class, reuse=False)

                    fix_hypothesis = Model.original(X_A[gpu_id])
                    fix_cost = tf.nn.softmax_cross_entropy_with_logits(
                        logits=fix_hypothesis,
                        labels=Y_A[gpu_id])

                    fix_losses.append(fix_cost)                      

                    fix_correct_prediction = tf.equal(tf.argmax(fix_hypothesis, 1), tf.arg_max(Y_A[gpu_id], 1))
                    fix_correct.append(fix_correct_prediction)

    fix_loss = tf.reduce_mean(tf.concat(fix_losses, axis=0))

    fix_accuracy = tf.reduce_mean(tf.cast(fix_correct, tf.float32) , name='fix_accuracy')
    fix_acc_sum = tf.summary.scalar("fix_accuarcy", fix_accuracy)

    var_all = tf.trainable_variables(scope=None)

    return (fix_loss), (fix_accuracy), (var_all)

def make_optimizer(placeholder, loss, var_list):
    _, _, _, learning_rate = placeholder

    fix_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, 
                                            var_list=var_list, colocate_gradients_with_ops=True)

    return fix_optimizer

def fit_model(sess, optimizer, placeholder, iterator, train_ds, step, FLAGS, log_manager, summary_opreation=None, writer=None):
    # From article: We trained our models using stochastic gradient descent with a batch size of 128 examples.
    X, Y, dropout, learning_rate = placeholder
    merged_summary = summary_opreation
    batch_size = FLAGS.batch_size * FLAGS.gpu_nums
    index = 0
    initial_lr = FLAGS.learning_rate
    lr = initial_lr
    dropout_rate = FLAGS.dropout

    if iterator is not None:
        sess.run(iterator.initializer)

    if step >= 55:
        lr = lr / 5
        if step >= 150:
            lr = lr / 10
            if step >= 180:
                lr = lr / 10

    if 'CIFAR' in FLAGS.dataset:
        X_data = []
        Y_data = []
        
        X_data, Y_data = train_ds
        num_examples = len(X_data)
        
        for offset in range(0, num_examples, batch_size):
            # sys.stdout.write("\r %d / %d" % (int(offset), num_examples))
            # sys.stdout.flush()

            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            
            batch_x, batch_y = next_batch(batch_size, batch_x, batch_y)
            batch_x = data_augmentation(batch_x)

            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: batch_x, Y: batch_y, dropout:dropout_rate, learning_rate:lr})
            writer.add_summary(summary, step * (num_examples // batch_size + 1) + index)
            index += 1
            log_manager.print_train_result()
    else:
        if 'imagenet' in FLAGS.dataset:
            num_examples = 1281167
        elif FLAGS.dataset == 'fashion_mnist' or FLAGS.dataset == 'mnist':
            num_examples = 60000
        else:
            num_examples = 50000
        for index in range(num_examples//batch_size):
            sys.stdout.write("\r %d / %d" % (index, (num_examples//batch_size)))
            sys.stdout.flush()

            image, label = sess.run(train_ds)
            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: image, Y:label, dropout:1.0})
            writer.add_summary(summary, step * (num_examples//batch_size) + index)

def train_validate(sess, accuracy, iterator, placeholder, test_ds, FLAGS, logHandler):
    X, Y, dropout, _ = placeholder
    batch_size = FLAGS.batch_size * FLAGS.gpu_nums
    total_accuracy = 0
    if iterator is not None:
        sess.run(iterator.initializer)
    if 'CIFAR' in FLAGS.dataset:
        X_data = []
        Y_data = []

        X_data, Y_data = test_ds
        num_examples = len(X_data)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            # batch_x, batch_y = next_batch(batch_size, batch_x, batch_y)
            
            batch_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout:1.0})
            total_accuracy += (batch_accuracy * len(batch_x))
        total_accuracy = total_accuracy / num_examples
    else:
        if 'imagenet' in FLAGS.dataset:
            num_examples = 1281167
        elif FLAGS.dataset == 'fashion_mnist' or FLAGS.dataset == 'mnist':
            num_examples = 60000
        else:
            num_examples = 50000

        for index in range(int(num_examples/batch_size)):

            image, label = sess.run(test_ds)

            batch_accuracy = sess.run(accuracy, feed_dict={X:image, Y:label, dropout:1.0})
            total_accuracy += (batch_accuracy * len(image))
        total_accuracy = total_accuracy / num_examples
    logHandler.print_comprehensive(total_accuracy, train_mode=True)
    return total_accuracy

def test_validate(sess, accuracy, iterator, placeholder, test_ds, FLAGS, logHandler):
    X, Y, dropout, _ = placeholder
    batch_size = FLAGS.batch_size * FLAGS.gpu_nums
    total_accuracy = 0
    if iterator is not None:
        sess.run(iterator.initializer)
    if 'CIFAR' in FLAGS.dataset:
        X_data = []
        Y_data = []

        X_data, Y_data = test_ds
        num_examples = len(X_data)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            # batch_x, batch_y = next_batch(batch_size, batch_x, batch_y)
            # batch_x = color_normalizing(batch_x)
            
            batch_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout:1.0})
            total_accuracy += (batch_accuracy * len(batch_x))
        total_accuracy = total_accuracy / num_examples
    else:
        if 'imagenet' in FLAGS.dataset:
            num_examples = 50000
        elif FLAGS.dataset == 'fashion_mnist' or FLAGS.dataset == 'mnist':
            num_examples = 10000
        else:
            num_examples = 10000

        for index in range(int(num_examples/batch_size)):

            image, label = sess.run(test_ds)

            batch_accuracy = sess.run(accuracy, feed_dict={X:image, Y:label, dropout:1.0})
            total_accuracy += (batch_accuracy * len(image))
        total_accuracy = total_accuracy / num_examples
    # logHandler.print_comprehensive(total_accuracy, train_mode=False)
    # logHandler._print('')
    return total_accuracy
