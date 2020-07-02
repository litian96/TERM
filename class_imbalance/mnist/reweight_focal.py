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
# Models for MNIST experiments.
#
from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def get_model(inputs,
              labels,
              alpha,
              gamma,
              is_training=True,
              is_tilting=False,
              dtype=tf.float32,
              w_dict=None,
              reuse=None):
    """Builds a simple LeNet.

    :param inputs:            [Tensor]    Inputs.
    :param labels:            [Tensor]    Labels.
    :param is_training:       [bool]      Whether in training mode, default True.
    :param dtype:             [dtype]     Data type, default tf.float32.
    :param w_dict:            [dict]      Dictionary of weights, default None.
    :param ex_wts:            [Tensor]    Example weights placeholder, default None.
    :param reuse:             [bool]      Whether to reuse variables, default None.
    """

    if w_dict is None:
        w_dict = {}

    def _get_var(name, shape, dtype, initializer):
        key = tf.get_variable_scope().name + '/' + name
        if key in w_dict:
            return w_dict[key]
        else:
            var = tf.get_variable(name, shape, dtype, initializer=initializer)
            w_dict[key] = var
            return var

    with tf.variable_scope('Model', reuse=reuse):
        inputs_ = tf.cast(tf.reshape(inputs, [-1, 28, 28, 1]), dtype)
        labels = tf.cast(labels, dtype)

        w_init = tf.truncated_normal_initializer(stddev=0.1)
        w1 = _get_var('w1', [5, 5, 1, 16], dtype, initializer=w_init)  # [14, 14, 16]
        w2 = _get_var('w2', [5, 5, 16, 32], dtype, initializer=w_init)  # [7, 7, 32]
        w3 = _get_var('w3', [5, 5, 32, 64], dtype, initializer=w_init)  # [4, 4, 64]
        w4 = _get_var('w4', [1024, 100], dtype, initializer=w_init)
        w5 = _get_var('w5', [100, 1], dtype, initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b1 = _get_var('b1', [16], dtype, initializer=b_init)
        b2 = _get_var('b2', [32], dtype, initializer=b_init)
        b3 = _get_var('b3', [64], dtype, initializer=b_init)
        b4 = _get_var('b4', [100], dtype, initializer=b_init)
        b5 = _get_var('b5', [1], dtype, initializer=b_init)

        act = tf.nn.relu

        # Conv-1
        l0 = tf.identity(inputs_, name='l0')
        z1 = tf.add(tf.nn.conv2d(inputs_, w1, [1, 1, 1, 1], 'SAME'), b1, name='z1')
        l1 = act(tf.nn.max_pool(z1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l1')

        # Conv-2
        z2 = tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2, name='z2')
        l2 = act(tf.nn.max_pool(z2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l2')

        # Conv-3
        z3 = tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3, name='z3')
        l3 = act(tf.nn.max_pool(z3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l3')

        # FC-4
        z4 = tf.add(tf.matmul(tf.reshape(l3, [-1, 1024]), w4), b4, name='z4')
        l4 = act(z4, name='l4')

        # FC-5
        z5 = tf.add(tf.matmul(l4, w5), b5, name='z5')

        logits = tf.squeeze(z5)
        out = tf.sigmoid(logits)

        y_true = tf.cast(labels, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * out + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - out)

        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.log(p_t)

        loss = tf.reduce_mean(focal_loss)

    return w_dict, loss, logits

