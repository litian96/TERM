
import sys
import numpy as np
import tensorflow as tf
import time
import inception_model
import cifar_data_provider
import tensorflow.contrib.slim as slim

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

flags.DEFINE_string('master', None, 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('data_dir', '', 'Data dir')

flags.DEFINE_string('train_log_dir', '', 'Directory to the save trained model.')

flags.DEFINE_string('dataset_name', 'cifar10', 'cifar10 or cifar100')

flags.DEFINE_string('studentnet', 'resnet101', 'inception or resnet101')

flags.DEFINE_float('learning_rate', 0.1, 'The learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'learning rate decay factor.')

flags.DEFINE_float('num_epochs_per_decay', 50,
                   'Number of epochs after which learning rate decays.')

flags.DEFINE_float('tilting', -0.1,
                   't in the tilting objective')

flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer(
    'save_interval_secs', 1200,
    'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('max_number_of_steps', 39000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_string('device_id', '0', 'GPU device ID to run the job.')

FLAGS = flags.FLAGS

# turn this on if there are no log outputs
tf.logging.set_verbosity(tf.logging.INFO)

with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    tf_global_step = tf.train.get_or_create_global_step()

    images, one_hot_labels, num_samples_per_epoch, num_of_classes = cifar_data_provider.provide_cifarnet_data(
        FLAGS.dataset_name,
        'train',
        FLAGS.batch_size,
        dataset_dir=FLAGS.data_dir)

    with slim.arg_scope(
            inception_model.cifarnet_arg_scope(weight_decay=0.004)):
        logits, _ = inception_model.cifarnet(images, num_of_classes, is_training=True, dropout_keep_prob=0.8)

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)

    loss = tf.reshape(total_loss, [-1, 1])
    total_loss = (1.0 / FLAGS.tilting) * tf.log(
        (1.0 / FLAGS.batch_size) * tf.reduce_sum(tf.exp(FLAGS.tilting * loss)) + 1e-6)

    # Using latest tensorflow ProtoBuf.
    tf.contrib.deprecated.scalar_summary('Total Loss', total_loss)

    decay_steps = int(
        num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    lr = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf_global_step,
        decay_steps,
        FLAGS.learning_rate_decay_factor,
        staircase=True)
    slim.summaries.add_scalar_summary(lr, 'learning_rate', print_summary=True)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, _ = zip(*grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(200000):
    print('step', i)
    sess.run(train_op)
