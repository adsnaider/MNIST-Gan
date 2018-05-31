from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from absl import app
from absl import flags
from absl import logging

from model import Discriminator, Generator

from buffer import RingBuffer

logging.set_verbosity(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', 'input/digits.tfrecords', 'Path to real data')

flags.DEFINE_float('generator_learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
flags.DEFINE_integer('iterations', 150000, 'Number of trainig iterations')
flags.DEFINE_integer('seed', 971,
                     'Seed to feed the tensorflow graph and numpy state')
flags.DEFINE_integer('D_iters', 5,
                     'Number of iterations of discriminator before switching')
flags.DEFINE_integer('G_iters', 2,
                     'Number of iterations of generator before switching')
flags.DEFINE_integer('running_average_count', 32,
                     'Number of samples to smooth out losses')

flags.DEFINE_string('checkpoint_directory', 'checkpoints/',
                    'Directory to read and write checkpoints')
flags.DEFINE_string('summary_directory', 'summaries/',
                    'Directory to write summaries')
flags.DEFINE_bool('save_checkpoints', True, 'Whether to save checkpoints')
flags.DEFINE_bool('save_summaries', True, 'Whether to save summaries')
flags.DEFINE_bool('restore', True, 'Whether to restore checkpoints (if exists)')
flags.DEFINE_integer('checkpoint_save_secs', 120,
                     'How often to save checkpoints')
flags.DEFINE_integer('summary_save_secs', None, 'How often to save summaries')
flags.DEFINE_integer('checkpoint_save_steps', None,
                     'How often to save checkpoints')
flags.DEFINE_integer('summary_save_steps', 100, 'How often to save summaries')
flags.DEFINE_integer('log_step', 100, 'How often to write console logs')


def _parse_record(example):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed_feature = tf.parse_single_example(example, features)
    image = parsed_feature['image']
    image = tf.decode_raw(image, tf.int64)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32)
    image = image / 255    # Normalize

    label = parsed_feature['label']
    label = tf.one_hot(label, 10)

    return image, label


def main(argv):
    del argv
    config = FLAGS

    tf.set_random_seed(config.seed)
    np_state = np.random.RandomState(config.seed)

    global_step = tf.train.get_or_create_global_step()
    global_step_update = tf.assign(global_step, global_step + 1)

    real_ds = tf.data.TFRecordDataset(config.input_path)
    real_ds = real_ds.map(_parse_record)
    real_ds = real_ds.shuffle(buffer_size=1000)
    real_ds = real_ds.batch(config.batch_size // 2)    # Half will be generated
    real_ds = real_ds.repeat()
    real_ds_iterator = real_ds.make_one_shot_iterator()
    real_ds_example, real_ds_label = real_ds_iterator.get_next()

    discriminator = Discriminator('discriminator')
    generator = Generator('generator')

    z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
    z_label = tf.placeholder(dtype=tf.int32, shape=[None])
    z_hot_label = tf.one_hot(z_label, 10)

    G_sample = generator.create_main_graph(z, z_hot_label)

    D_logit_real = discriminator.create_main_graph(real_ds_example,
                                                   real_ds_label)
    D_logit_fake = discriminator.create_main_graph(G_sample, z_hot_label)

    D_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(D_logit_real), D_logit_real, label_smoothing=0.2)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(D_logit_fake), D_logit_fake, label_smoothing=0.00)

    D_loss = 0.5 * (D_loss_real + D_loss_fake)

    G_loss = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(D_logit_fake), D_logit_fake, label_smoothing=0.00)

    update_ops = tf.get_collection(
        tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    with tf.control_dependencies(update_ops):
        D_optimizer = tf.train.AdamOptimizer(
            config.discriminator_learning_rate).minimize(
                D_loss, var_list=discriminator.get_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
        G_optimizer = tf.train.AdamOptimizer(
            config.generator_learning_rate).minimize(
                G_loss, var_list=generator.get_variables())

    with tf.variable_scope('summaries'):
        D_loss_summary = tf.summary.scalar(
            'loss', D_loss, family='discriminator')
        G_loss_summary = tf.summary.scalar('loss', G_loss, family='generator')
        G_image_summary = tf.summary.image(
            'generation', G_sample, max_outputs=1, family='generator')
        Real_image_summary = tf.summary.image(
            'real', real_ds_example, max_outputs=1)

        summary_op = tf.summary.merge_all()

    # Session
    hooks = []
    hooks.append(tf.train.StopAtStepHook(num_steps=config.iterations))
    if (config.save_checkpoints):
        hooks.append(
            tf.train.CheckpointSaverHook(
                checkpoint_dir=config.checkpoint_directory,
                save_secs=config.checkpoint_save_secs,
                save_steps=config.checkpoint_save_steps))

    if (config.save_summaries):
        hooks.append(
            tf.train.SummarySaverHook(
                output_dir=config.summary_directory,
                save_secs=config.summary_save_secs,
                save_steps=config.summary_save_steps,
                summary_op=summary_op))

    if config.restore:
        sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=config.checkpoint_directory,
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            log_step_count_steps=None,
            hooks=hooks)
    else:
        sess = tf.train.MonitoredTrainingSession(
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            log_step_count_steps=None,
            hooks=hooks)

    tf.get_default_graph().finalize()

    def step_generator(step_context):
        np_global_step = step_context.session.run(global_step)
        step_context.session.run(global_step_update)

        random_noise = np_state.normal(size=[config.batch_size, 100])
        random_label = np_state.randint(10, size=config.batch_size)
        _, np_loss = step_context.run_with_hooks(
            [G_optimizer, G_loss],
            feed_dict={
                z: random_noise,
                z_label: random_label
            })

        if np_global_step % config.log_step == 0:
            logging.debug('Training Generator: Step: {}   Loss: {:.3e}'.format(
                np_global_step, np_loss))

    def step_discriminator(step_context):
        np_global_step = step_context.session.run(global_step)
        step_context.session.run(global_step_update)

        random_noise = np_state.normal(size=[config.batch_size // 2, 100])
        random_label = np_state.randint(10, size=config.batch_size // 2)
        _, np_loss = step_context.run_with_hooks(
            [D_optimizer, D_loss],
            feed_dict={
                z: random_noise,
                z_label: random_label
            })

        if np_global_step % config.log_step == 0:
            logging.debug(
                'Training Discriminator: Step: {}   Loss Mean: {:.3e}'.format(
                    np_global_step, np_loss))

    while not sess.should_stop():
        should_run_generator = np_state.randint(
            config.G_iters + config.D_iters) < config.G_iters
        try:
            if should_run_generator:
                sess.run_step_fn(
                    lambda step_context: step_generator(step_context))
            else:
                sess.run_step_fn(
                    lambda step_context: step_discriminator(step_context))
        except tf.errors.OutOfRangeError:
            break

    sess.close()


if __name__ == '__main__':
    app.run(main)
