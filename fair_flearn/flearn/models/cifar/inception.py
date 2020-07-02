import numpy as np
import tensorflow as tf
from tqdm import trange
import tensorflow.contrib.slim as slim

from flearn.utils.model_utils import batch_data, gen_batch
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

def process_x(raw_x_batch):
    x_batch = np.array(raw_x_batch)
    return x_batch

def process_y(raw_y_batch):
    labels = np.array(raw_y_batch)
    y_batch = np.eye(10)[labels]
    return y_batch

class Model(object):
    def __init__(self, num_classes, q, optimizer, seed):
        self.num_classes = num_classes

        self.graph = tf.Graph()
        with self.graph.as_default():
            # tf.set_random_seed(123 + seed)
            tf.set_random_seed(456 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                optimizer, q)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer, q, dropout_keep_prob=0.5, prediction_fn=slim.softmax):
        end_points = {}
        images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        one_hot_labels = tf.placeholder(tf.int32, [None, self.num_classes])
        net = slim.conv2d(images, 64, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 384, scope='fc3')
        net = slim.fully_connected(net, 192, scope='fc4')
        logits = slim.fully_connected(net, self.num_classes,
                                      biases_initializer=tf.zeros_initializer(),
                                      weights_initializer=trunc_normal(1 / 192.0),
                                      weights_regularizer=None,
                                      activation_fn=None,
                                      scope='logits')
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot_labels, logits=logits)

        #loss = tf.reduce_mean(loss)

        loss = tf.reshape(loss, [-1, 1])

        loss = (1.0 / q) * tf.log(
            (1.0 / 128) * tf.reduce_sum(tf.exp(q * loss)) + 1e-6)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return images, one_hot_labels, train_op, grads, eval_metric_ops, loss

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

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss,
                                 feed_dict={self.features: process_x(data['x']), self.labels: process_y(data['y'])})
        return loss

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            input_data = process_x(mini_batch_data[0])
            target_data = process_y(mini_batch_data[1])
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                    feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        return grads, loss, soln

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            soln: trainable variables of the lstm model
            comp: number of FLOPs computed while training given data
        '''
        for idx in trange(num_epochs, desc='Epoch: ', leave=False):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp


    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            tot_correct: total #samples that are predicted correctly
            loss: loss value on `data`
        '''
        l = 0
        tot = 0
        for X, y in batch_data(data, 125):
            with self.graph.as_default():
                tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: process_x(X), self.labels: process_y(y)})
                l += loss
                tot += tot_correct
        return tot, l / 80

    def close(self):
        self.sess.close()

