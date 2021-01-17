# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Runs MNIST experitment. Default 10 runs for 10 random seeds.
#
#
# Flags:
# --exp             [string]         Experiment name, `tilting`, `learning`, `hm`, `ratio`, `random` or `baseline`.
# --pos_ratio       [float]          The ratio for the positive class, choose between 0.9 - 0.995.
# --nrun            [int]            Total number of runs with different random seeds.
# --ntrain          [int]            Number of training examples.
# --nval            [int]            Number of validation examples.
# --ntest           [int]            Number of test examples.
# --tensorboard                      Writes TensorBoard logs while training, default True.
# --notensorboard                    Disable TensorBoard.
# --verbose                          Print training progress, default False.
# --noverbose                        Disable printing.
#

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import six
import tensorflow as tf
import math

from collections import namedtuple
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

from mnist.reweight import get_model, reweight_random, reweight_autodiff, reweight_hard_mining, reweight_tilting
from utils.logger import get as get_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.flags
flags.DEFINE_float('pos_ratio', 0.98, 'Ratio of positive examples in training')
flags.DEFINE_integer('nrun', 5, 'Number of runs')
flags.DEFINE_integer('ntest', 500, 'Number of testing examples')
flags.DEFINE_integer('ntrain', 5000, 'Number of training examples')
flags.DEFINE_integer('nval', 10, 'Number of validation examples')
flags.DEFINE_bool('verbose', True, 'Whether to print training progress')
flags.DEFINE_bool('tensorboard', True, 'Whether to save training progress')
flags.DEFINE_string('exp', 'baseline', 'Which experiment to run')
FLAGS = tf.flags.FLAGS

log = get_logger()

Config = namedtuple('Config', [
    'reweight', 'lr', 'num_steps', 'random', 'ratio_weighted', 'nval', 'hard_mining', 'bsize',  'learning', 'tilting'
])

exp_repo = dict()


def RegisterExp(name):
    def _decorator(f):
        exp_repo[name] = f
        return f

    return _decorator


LR = 0.002
NUM_STEPS = 2000


@RegisterExp('baseline')
def baseline_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=False,
        tilting=False,
        learning=False,
        bsize=100,
        nval=0)


@RegisterExp('hm')
def baseline_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=True,
        tilting=False,
        learning=False,
        bsize=500,
        nval=0)


@RegisterExp('ratio')
def ratio_config():
    return Config(
        reweight=False,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=False,
        ratio_weighted=True,
        hard_mining=False,
        bsize=100,
        tilting=False,
        learning=False,
        nval=0)


@RegisterExp('random')
def dpfish_config():
    return Config(
        reweight=True,
        num_steps=NUM_STEPS * 2,
        lr=LR,
        random=True,
        ratio_weighted=False,
        hard_mining=False,
        learning=False,
        bsize=100,
        tilting=False,
        nval=0)


@RegisterExp('learning')
def learning_config():
    return Config(
        reweight=True,
        num_steps=NUM_STEPS,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=False,
        learning=True,
        tilting=False,
        bsize=100,
        nval=FLAGS.nval)

@RegisterExp('tilting')
def tilting_config():
    return Config(
        reweight=True,
        num_steps=NUM_STEPS,
        lr=LR,
        random=False,
        ratio_weighted=False,
        hard_mining=False,
        learning=False,
        tilting=True,
        bsize=500,
        nval=0)


def get_imbalance_dataset(mnist,
                          pos_ratio=0.9,
                          ntrain=5000,
                          nval=10,
                          ntest=500,
                          seed=0,
                          class_0=4,
                          class_1=9):
    rnd = np.random.RandomState(seed)

    # In training, we have 10% 4 and 90% 9.
    # In testing, we have 50% 4 and 50% 9.
    ratio = 1 - pos_ratio
    ratio_test = 0.5

    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    x_train_0 = x_train[y_train == class_0]
    x_test_0 = x_test[y_test == class_0]

    # First shuffle, negative.
    idx = np.arange(x_train_0.shape[0])
    rnd.shuffle(idx)
    x_train_0 = x_train_0[idx]

    nval_small_neg = int(np.floor(nval * ratio_test))
    ntrain_small_neg = int(np.floor(ntrain * ratio)) - nval_small_neg

    x_val_0 = x_train_0[:nval_small_neg]
    x_train_0 = x_train_0[nval_small_neg:nval_small_neg + ntrain_small_neg]

    if FLAGS.verbose:
        print('Number of train negative classes', ntrain_small_neg)
        print('Number of val negative classes', nval_small_neg)

    idx = np.arange(x_test_0.shape[0])
    rnd.shuffle(idx)
    x_test_0 = x_test_0[:int(np.floor(ntest * ratio_test))]

    x_train_1 = x_train[y_train == class_1]
    x_test_1 = x_test[y_test == class_1]

    # First shuffle, positive.
    idx = np.arange(x_train_1.shape[0])
    rnd.shuffle(idx)
    x_train_1 = x_train_1[idx]

    nvalsmall_pos = int(np.floor(nval * (1 - ratio_test)))
    ntrainsmall_pos = int(np.floor(ntrain * (1 - ratio))) - nvalsmall_pos

    x_val_1 = x_train_1[:nvalsmall_pos]    # 50 9 in validation.
    x_train_1 = x_train_1[nvalsmall_pos:nvalsmall_pos + ntrainsmall_pos]    # 4500 9 in training.

    idx = np.arange(x_test_1.shape[0])
    rnd.shuffle(idx)
    x_test_1 = x_test_1[idx]
    x_test_1 = x_test_1[:int(np.floor(ntest * (1 - ratio_test)))]    # 500 9 in testing.

    if FLAGS.verbose:
        print('Number of train positive classes', ntrainsmall_pos)
        print('Number of val positive classes', nvalsmall_pos)

    y_train_subset = np.concatenate([np.zeros([x_train_0.shape[0]]), np.ones([x_train_1.shape[0]])])
    y_val_subset = np.concatenate([np.zeros([x_val_0.shape[0]]), np.ones([x_val_1.shape[0]])])
    y_test_subset = np.concatenate([np.zeros([x_test_0.shape[0]]), np.ones([x_test_1.shape[0]])])

    y_train_pos_subset = np.ones([x_train_1.shape[0]])
    y_train_neg_subset = np.zeros([x_train_0.shape[0]])

    x_train_subset = np.concatenate([x_train_0, x_train_1], axis=0).reshape([-1, 28, 28, 1])
    x_val_subset = np.concatenate([x_val_0, x_val_1], axis=0).reshape([-1, 28, 28, 1])
    x_test_subset = np.concatenate([x_test_0, x_test_1], axis=0).reshape([-1, 28, 28, 1])

    x_train_pos_subset = x_train_1.reshape([-1, 28, 28, 1])
    x_train_neg_subset = x_train_0.reshape([-1, 28, 28, 1])

    # Final shuffle.
    idx = np.arange(x_train_subset.shape[0])
    rnd.shuffle(idx)
    x_train_subset = x_train_subset[idx]
    y_train_subset = y_train_subset[idx]

    idx = np.arange(x_val_subset.shape[0])
    rnd.shuffle(idx)
    x_val_subset = x_val_subset[idx]
    y_val_subset = y_val_subset[idx]

    idx = np.arange(x_test_subset.shape[0])
    rnd.shuffle(idx)
    x_test_subset = x_test_subset[idx]
    y_test_subset = y_test_subset[idx]

    train_set = DataSet(x_train_subset * 255.0, y_train_subset)
    train_pos_set = DataSet(x_train_pos_subset * 255.0, y_train_pos_subset)
    train_neg_set = DataSet(x_train_neg_subset * 255.0, y_train_neg_subset)
    val_set = DataSet(x_val_subset * 255.0, y_val_subset)
    test_set = DataSet(x_test_subset * 255.0, y_test_subset)

    return train_set, val_set, test_set, train_pos_set, train_neg_set


def get_exp_logger(sess, log_folder):
    """Gets a TensorBoard logger."""
    with tf.name_scope('Summary'):
        writer = tf.summary.FileWriter(os.path.join(log_folder), sess.graph)

    class ExperimentLogger():
        def log(self, niter, name, value):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=value)
            writer.add_summary(summary, niter)

        def flush(self):
            """Flushes results to disk."""
            writer.flush()

    return ExperimentLogger()


def evaluate(sess, x_, y_, acc_, train_set, test_set):
    # Calculate final results.
    acc_sum = 0.0
    acc_test_sum = 0.0
    train_bsize = 100
    for step in six.moves.xrange(5000 // train_bsize):
        x, y = train_set.next_batch(train_bsize)
        acc = sess.run(acc_, feed_dict={x_: x, y_: y})
        acc_sum += acc

    test_bsize = 100
    for step in six.moves.xrange(500 // test_bsize):
        x_test, y_test = test_set.next_batch(test_bsize)
        acc = sess.run(acc_, feed_dict={x_: x_test, y_: y_test})
        acc_test_sum += acc

    train_acc = acc_sum / float(5000 // train_bsize)
    test_acc = acc_test_sum / float(500 // test_bsize)
    return train_acc, test_acc


def get_acc(logits, y):
    prediction = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))


def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = np.random.uniform(0, 1, shape)
  return -np.log(-np.log(U + eps) + eps)

def run(dataset, exp_name, seed, verbose=True):
    pos_ratio = FLAGS.pos_ratio
    ntrain = FLAGS.ntrain
    nval = FLAGS.nval
    ntest = FLAGS.ntest
    folder = os.path.join('ckpt_mnist_imbalance_cnn_p{:d}'.format(int(FLAGS.pos_ratio * 100.0)),
                          exp_name + '_{:d}'.format(seed))
    if not os.path.exists(folder):
        os.makedirs(folder)

    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(123)
        config = exp_repo[exp_name]()
        bsize = config.bsize
        train_set, val_set, test_set, train_pos_set, train_neg_set = get_imbalance_dataset(
            dataset, pos_ratio=pos_ratio, ntrain=ntrain, nval=config.nval, ntest=ntest, seed=seed)

        x_ = tf.placeholder(tf.float32, [None, 784], name='x')
        y_ = tf.placeholder(tf.float32, [None], name='y')
        x_val_ = tf.placeholder(tf.float32, [None, 784], name='x_val')
        y_val_ = tf.placeholder(tf.float32, [None], name='y_val')
        ex_wts_ = tf.placeholder(tf.float32, [None], name='ex_wts')
        lr_ = tf.placeholder(tf.float32, [], name='lr')

        # Build training model.
        with tf.name_scope('Train'):
            _, loss_c, logits_c, individual_loss = get_model(
                x_, y_, is_training=True, dtype=tf.float32, w_dict=None, ex_wts=ex_wts_, reuse=None)
            train_op = tf.train.AdamOptimizer(lr_).minimize(loss_c)

        # Build evaluation model.
        with tf.name_scope('Val'):
            _, loss_eval, logits_eval, _ = get_model(
                x_,
                y_,
                is_training=False,
                dtype=tf.float32,
                w_dict=None,
                ex_wts=ex_wts_,
                reuse=True)
            acc_ = get_acc(logits_eval, y_)


        num_steps = config.num_steps

        sess.run(tf.global_variables_initializer())

        for step in six.moves.xrange(num_steps):
            loss_list = []
            train_x = []
            train_y = []
            for _ in range(int(5000/bsize)):
                x, y = train_set.next_batch(bsize)
                train_x.extend(x)
                train_y.extend(y)
                losses = sess.run(individual_loss, feed_dict={x_: x, y_: y, ex_wts_: np.ones(len(x))/len(x)})
                loss_list.extend(losses)
            loss_array = np.array(loss_list)

            train_x = np.array(train_x)
            train_y = np.array(train_y)
            positive_indices = np.where(train_y == 1)[0]
            negative_indices = np.where(train_y == 0)[0]
            loss_positive = np.mean(loss_array[positive_indices])
            loss_negative = np.mean(loss_array[negative_indices])
            max_l = max(loss_positive, loss_negative)

            tt = 100

            weights_positive = len_positive * np.exp(tt * (loss_positive-max_l)) / (
                        len_positive * np.exp(tt * (loss_positive-max_l)) + len_negative * np.exp(tt * (loss_negative-max_l)))
            weights_negative = len_negative * np.exp(tt * (loss_negative-max_l)) / (
                        len_positive * np.exp(tt * (loss_positive-max_l)) + len_negative * np.exp(tt * (loss_negative-max_l)))

            np.random.seed(seed + step)
            label_ = np.random.choice([1, 0], 1, p=np.array([weights_positive, weights_negative]))
            if label_ == 1:
                positive_batch = np.random.choice(positive_indices, 100)
                train_x_batch = train_x[positive_batch]
                train_y_batch = train_y[positive_batch]
            else:
                negative_batch = np.random.choice(negative_indices, 100)
                train_x_batch = train_x[negative_batch]
                train_y_batch = train_y[negative_batch]

            x_val, y_val = val_set.next_batch(min(bsize, nval))

            lr = config.lr

            loss, acc, _ = sess.run(
                [loss_c, acc_, train_op],
                feed_dict={
                    x_: train_x_batch,
                    y_: train_y_batch,
                    x_val_: x_val,
                    y_val_: y_val,
                    ex_wts_: np.ones(len(train_y_batch))/len(train_y_batch),
                    lr_: lr
                })
            if (step + 1) % 50 == 0:
                train_acc, test_acc = evaluate(sess, x_, y_, acc_, train_set, test_set)
                if verbose:
                    print('Step', step + 1, 'Loss', loss, 'Train acc', train_acc, 'Test acc',
                          test_acc)

                if loss < 0.01:
                    break

        # Final evaluation.
        train_acc, test_acc = evaluate(sess, x_, y_, acc_, train_set, test_set)
        if verbose:
            print('Final', 'Train acc', train_acc, 'Test acc', test_acc)
    return train_acc, test_acc


def run_many(dataset, exp_name):
    train_acc_list = []
    test_acc_list = []
    for trial in tqdm(six.moves.xrange(FLAGS.nrun), desc=exp_name):
        np.random.seed(12345+trial)
        train_acc, test_acc = run(
            dataset, exp_name, (trial * 123456789) % 100000, verbose=FLAGS.verbose)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    train_acc_list = np.array(train_acc_list)
    test_acc_list = np.array(test_acc_list)
    print(exp_name, 'Train acc {:.3f}% ({:.3f}%)'.format(train_acc_list.mean() * 100.0,
                                                         train_acc_list.std() * 100.0))
    print(exp_name, 'Test acc {:.3f}% ({:.3f}%)'.format(test_acc_list.mean() * 100.0,
                                                        test_acc_list.std() * 100.0))


def main():

    mnist = input_data.read_data_sets("data/mnist", one_hot=False)
    for exp in FLAGS.exp.split(','):
        run_many(mnist, exp)


if __name__ == '__main__':
    main()
