import numpy as np
import os, sys
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def relu_weight_variable(shape):
    assert len(shape) is 2
    input_size = shape[0]
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / input_size))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides):
    return conv_batch_normalization(tf.nn.conv2d(x, W, strides=strides, padding='SAME'))


def conv_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)


def fc_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, q, optimizer, loss, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                q, optimizer, loss)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, q, optimizer, loss):
        """Model function for CNN."""

        features = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if loss == 'unmodified':  # original cross entropy
            # logits must be one-hot encoding; labels must be just scalars
            # return the average loss
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        elif loss == 'ce':  # another implementation of cross entropy loss (exactly the same as unmodified)
            y_pred = predictions['probabilities']
            one_hot_labels = tf.one_hot(labels, depth=y_pred.shape[1])
            loss = -one_hot_labels * tf.log(y_pred)  # (batch size, classes)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        elif loss == 'fl':
            #alpha = tf.constant([0.05, 0.95])
            y_pred = predictions['probabilities']
            one_hot_labels = tf.one_hot(labels, depth=y_pred.shape[1])
            loss = -one_hot_labels * tf.log(y_pred) * ((1 - y_pred) ** 2)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        elif loss == 'qffl':
            y_pred = predictions['probabilities']
            one_hot_labels = tf.one_hot(labels, depth=y_pred.shape[1])
            loss = -one_hot_labels * tf.log(y_pred)
            loss = tf.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(tf.pow(loss, 2))

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, latest_model):

        '''
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return model_grads
        '''
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})
        weights = self.get_params()
        return grads, loss, weights

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
