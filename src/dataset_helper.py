# -*- coding: utf-8 -*-

# The CIFAR-10 dataset:
# https://www.cs.toronto.edu/~kriz/cifar.html

import pickle
import numpy as np
import scipy.misc
import pandas as pd
import random
# from imgaug import augmenters as iaa
import keras
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
import tensorflow as tf

def _random_crop(batch, crop_shape, padding=None):
  oshape = np.shape(batch[0])

  if padding:
    oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
  new_batch = []
  npad = ((padding, padding), (padding, padding), (0, 0))
  for i in range(len(batch)):
    new_batch.append(batch[i])
    if padding:
      new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                    nw:nw + crop_shape[1]]
  return new_batch


def _random_flip_leftright(batch):
  for i in range(len(batch)):
    if bool(random.getrandbits(1)):
        batch[i] = np.fliplr(batch[i])
  return batch


def color_preprocessing(x_train):
  x_train = x_train.astype('float32')

  x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
  x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
  x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

  return x_train

def color_normalizing(x_train):
  x_train = x_train.astype('float32')

  x_train[:, :, :, 0] = (x_train[:, :, :, 0])/255.
  x_train[:, :, :, 1] = (x_train[:, :, :, 1])/255.
  x_train[:, :, :, 2] = (x_train[:, :, :, 2])/255.

  return x_train

def data_augmentation(batch):
  batch = _random_flip_leftright(batch)
  batch = _random_crop(batch, [32, 32], 4)
  # batch = color_normalizing(batch)
  return batch


def __unpickle(file):
  with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
  return dict

def read_cifar_10(image_width, image_height):
  batch_1 = __unpickle('/workspace/cifar-10/data_batch_1')
  batch_2 = __unpickle('/workspace/cifar-10/data_batch_2')
  batch_3 = __unpickle('/workspace/cifar-10/data_batch_3')
  batch_4 = __unpickle('/workspace/cifar-10/data_batch_4')
  batch_5 = __unpickle('/workspace/cifar-10/data_batch_5')
  test_batch = __unpickle('/workspace/cifar-10/test_batch')

  classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  total_train_samples = len(batch_1[b'labels']) + len(batch_2[b'labels']) + len(batch_3[b'labels'])\
                        + len(batch_4[b'labels']) + len(batch_5[b'labels'])

  X_train = np.zeros(shape=[total_train_samples, image_width, image_height, 3], dtype=np.uint8)
  Y_train = np.zeros(shape=[total_train_samples, len(classes)], dtype=np.float32)

  batches = [batch_1, batch_2, batch_3, batch_4, batch_5]

  index = 0
  for batch in batches:
    for i in range(len(batch[b'labels'])):
      image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
      label = batch[b'labels'][i]

      X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
      Y = np.zeros(shape=[len(classes)], dtype=np.int)
      Y[label] = 1

      X_train[index + i] = X
      Y_train[index + i] = Y

    index += len(batch[b'labels'])

  total_test_samples = len(test_batch[b'labels'])

  X_test = np.zeros(shape=[total_test_samples, image_width, image_height, 3], dtype=np.uint8)
  Y_test = np.zeros(shape=[total_test_samples, len(classes)], dtype=np.float32)

  for i in range(len(test_batch[b'labels'])):
    image = test_batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
    label = test_batch[b'labels'][i]

    X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
    Y = np.zeros(shape=[len(classes)], dtype=np.int)
    Y[label] = 1

    X_test[i] = X
    Y_test[i] = Y

  # print(Y_train[1])

  return X_train, Y_train, X_test, Y_test

def cifar100_load(image_width, image_height):
  train = __unpickle('/workspace/cifar-100-python/train')
  test = __unpickle('/workspace/cifar-100-python/test')
  meta = __unpickle('/workspace/cifar-100-python/meta')

  classes = 100
  total_train_samples = 50000
  total_test_samples = 10000

  X_train = np.zeros(shape=[total_train_samples, image_width, image_height, 3], dtype=np.uint8)
  Y_train = np.zeros(shape=[total_train_samples, classes], dtype=np.float32)

  index = 0

  for i in range(len(train[b'fine_labels'])):
    image = train[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
    label = train[b'fine_labels'][i]

    X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
    Y = np.zeros(shape=[classes], dtype=np.int)
    Y[label] = 1

    X_train[i] = X
    Y_train[i] = Y

  X_test = np.zeros(shape=[total_test_samples, image_width, image_height, 3], dtype=np.uint8)
  Y_test = np.zeros(shape=[total_test_samples, classes], dtype=np.float32)

  for i in range(len(test[b'fine_labels'])):
    image = test[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
    label = test[b'fine_labels'][i]

    X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
    Y = np.zeros(shape=[classes], dtype=np.int)
    Y[label] = 1

    X_test[i] = X
    Y_test[i] = Y

  # print(Y_train[1])

  return X_train, Y_train, X_test, Y_test



def check_available_gpus():
  local_devices = device_lib.list_local_devices()
  gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
  gpu_num = len(gpu_names)

  print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

  return gpu_num

def cifar_100_imgaug(image_width, image_height):
  train = __unpickle('/workspace/cifar-100-python/train')
  test = __unpickle('/workspace/cifar-100-python/test')
  num_class = 100
  X_save = np.zeros(shape=[500000, image_width, image_height, 3], dtype=np.uint8)

  train_list = []
  X_save = np.load('/workspace/cifar100-augmentation/IMG_AUG.npy')
  train_list.extend(X_save)

  labels_list = []
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  labels_list.extend(train[b'fine_labels'])
  
  train_shuffled = []
  labels_shuffled = []

  combined = list(zip(train_list, labels_list))
  random.shuffle(combined)

  train_shuffled[:], labels_shuffled[:] = zip(*combined)

  train_shuffled = np.asarray(train_shuffled)

  def one_hot_encode(vec, vals=num_class):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

  labels_shuffled = one_hot_encode(labels_shuffled, num_class)

  test_shuffled = np.vstack(test[b"data"])
  test_len = len(test_shuffled)

  test_shuffled = test_shuffled.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
  test_len = len(test_shuffled)
  test_labels = one_hot_encode(test[b'fine_labels'], num_class)

  return train_shuffled, labels_shuffled, test_shuffled, test_labels


def get_dataset(FLAGS):
  print("Data reading...")
  if 'CIFAR' in FLAGS.dataset:

    X = tf.placeholder(tf.float32, [None, 32, 32, 3]) 
    dropout_rate = tf.placeholder("float")
    learning_rate = tf.placeholder("float")

    if 'CIFAR100' == FLAGS.dataset:
      Y = tf.placeholder(tf.float32, [None, 100])
      X_train, Y_train, X_test, Y_test = cifar_100_imgaug(image_width=32, image_height=32)

    else:
      Y = tf.placeholder(tf.float32, [None, 10])
      X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=32, image_height=32)

  return (X_train, Y_train), (X_test, Y_test), (X, Y, dropout_rate, learning_rate)
  
