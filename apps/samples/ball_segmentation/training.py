'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
"""Segmentation training script.

Trains a neural network which takes in an rgb image as input and outputs a semantic segmentation
of a ball. The network architecture used is Unet. This script is setup to stream images from Isaac
SDK for training.
"""

import os
import glob
import PIL.Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Limit the GPU usage to gpu #0

from engine.pyalice import *
import apps.samples.ball_segmentation
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Input, MaxPooling2D
import keras.regularizers as regularizers
import packages.ml
import numpy as np
import shutil
import tensorflow as tf
import time

from tensorflow.python.tools import freeze_graph

# Command line flags:
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_logdir', '/tmp/ball_segmentation/',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('validation_datadir', 'apps/samples/ball_segmentation/ball_validation_dataset',
                    'Where the validation dataset is stored.')
flags.DEFINE_integer('training_steps', 10000, 'number of batches to train on')
flags.DEFINE_integer('summary_every', 10, 'write summary every x steps. x = 10 by default')
flags.DEFINE_integer('save_every', 1000, 'save the model for every x steps. x = 1000 by default')
flags.DEFINE_integer('batch_size', 8, 'batch size used for training')
flags.DEFINE_float('learning_rate', 1e-5, 'learning rate used for Adam optimizer')
flags.DEFINE_float('ball_class_weight', 25, 'Weighting of ball pixels in the loss relative to ' \
                   'background pixels')

flags.DEFINE_string('app_filename', "apps/samples/ball_segmentation/training.app.json",
                    'Where the isaac SDK app is stored')

flags.DEFINE_string(
    'checkpoint', "", "Default to not loading checkpoints. If it is specified, \
    the checkpoint will be loaded")

flags.DEFINE_float('gpu_memory_usage', 0.33, "Spceified to limit the usage of gpu")

# Op names.
COLOR_IMAGE_NAME = 'rgb_image'
LABEL_IMAGE_NAME = 'sgementation_label'
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

# Number of samples to acquire in batch
kSampleNumbers = 500


def get_generator(bridge):
    """Create a training sample generator.

    Args:
      bridge: the isaac sample accumulator node which we will acquire samples from

    Returns:
      A generator function which yields a single training example.
  """

    def _generator():
        # Indefinately yield samples.

        # Generator will stop if no samples are received from bridge for more than this number of
        # seconds.
        kFailIfBridgeStallsTime = 3    # in seconds
        # Amount of time generator will wait if bridge contains no samples
        kSleepOnBridgeStallTime = 1    # in seconds

        missing_sample_count = 0
        while True:
            # Try to acquire samples.
            samples = bridge.acquire_samples(kSampleNumbers)
            if not samples:
                missing_sample_count += 1
                if (missing_sample_count >= (kFailIfBridgeStallsTime / kSleepOnBridgeStallTime)):
                    # The timeout has passed, assume bridge has stalled and stop the generator.
                    raise StopIteration
                else:
                    # Wait for a bit so we do not spam the app, then try again.
                    time.sleep(kSleepOnBridgeStallTime)
                    continue

            # We received samples, reset missing sample count.
            missing_sample_count = 0

            # We should get at least one sample with two tensors (the input image and labels).
            assert len(samples) >= 1
            assert len(samples[0]) == 2

            for image, label in samples:
                yield {COLOR_IMAGE_NAME: image, LABEL_IMAGE_NAME: label}

    return _generator


def get_dataset(bridge):
    """Create a tf.data dataset which yields batches of samples for training.

  Args:
      bridge: the isaac sample accumulator node which we will acquire samples from

  Returns:
    A tf.data dataset which yields batches of training examples.
  """
    dataset = tf.data.Dataset.from_generator(
        get_generator(bridge), {
            COLOR_IMAGE_NAME: tf.float32,
            LABEL_IMAGE_NAME: tf.float32,
        }, {
            COLOR_IMAGE_NAME: (None, None, 3),
            LABEL_IMAGE_NAME: (None, None, 1),
        }).batch(FLAGS.batch_size)

    return dataset


def create_unet():
    network = {}
    l2_lambda = 0.03

    network["conv1_1"] = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv1_2"] = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["pool1"] = MaxPooling2D(pool_size=(2, 2))

    network["conv2_1"] = Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv2_2"] = Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["pool2"] = MaxPooling2D(pool_size=(2, 2))

    network["conv3_1"] = Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv3_2"] = Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["pool3"] = MaxPooling2D(pool_size=(2, 2))

    network["conv4_1"] = Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv4_2"] = Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["pool4"] = MaxPooling2D(pool_size=(2, 2))

    network["conv5_1"] = Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv5_2"] = Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))

    network["up6"] = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
    network["conv6_1"] = Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv6_2"] = Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))

    network["up7"] = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
    network["conv7_1"] = Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv7_2"] = Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))

    network["up8"] = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
    network["conv8_1"] = Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv8_2"] = Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))

    network["up9"] = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
    network["conv9_1"] = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    network["conv9_2"] = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))

    network["final_conv"] = Conv2D(
        1, (1, 1),
        kernel_regularizer=regularizers.l2(l2_lambda),
        bias_regularizer=regularizers.l2(l2_lambda))
    return network, []    #(vgg.layers[1].weights + vgg.layers[2].weights)


def unet(input, network):
    """Unet semantic segmentation network architecture.

    Code taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    Original paper: https://arxiv.org/abs/1505.04597

    Args:
      input: The input image, preprocessed so each pixel is in the range [-1, 1].

    Returns:
      A tuple containing the predicted ball segmentation and the logits (before the sigmoid).
  """
    inp = tf.identity(input, INPUT_NAME)

    conv1 = network["conv1_1"](inp)
    conv1 = network["conv1_2"](conv1)
    pool1 = network["pool1"](conv1)

    conv2 = network["conv2_1"](pool1)
    conv2 = network["conv2_2"](conv2)
    pool2 = network["pool2"](conv2)

    conv3 = network["conv3_1"](pool2)
    conv3 = network["conv3_2"](conv3)
    pool3 = network["pool3"](conv3)

    conv4 = network["conv4_1"](pool3)
    conv4 = network["conv4_2"](conv4)
    pool4 = network["pool3"](conv4)

    conv5 = network["conv5_1"](pool4)
    conv5 = network["conv5_2"](conv5)

    up6 = concatenate([network["up6"](conv5), conv4], axis=3)
    conv6 = network["conv6_1"](up6)
    conv6 = network["conv6_2"](conv6)

    up7 = concatenate([network["up7"](conv6), conv3], axis=3)
    conv7 = network["conv7_1"](up7)
    conv7 = network["conv7_2"](conv7)

    up8 = concatenate([network["up8"](conv7), conv2], axis=3)
    conv8 = network["conv8_1"](up8)
    conv8 = network["conv8_2"](conv8)

    up9 = concatenate([network["up9"](conv8), conv1], axis=3)
    conv9 = network["conv9_1"](up9)
    conv9 = network["conv9_2"](conv9)

    logits = network["final_conv"](conv9)
    predictions = tf.sigmoid(logits, name=OUTPUT_NAME)

    return predictions, logits


def make_dir(path):
    """Clears a directory and creates it on disk.

  Args:
    path: The path on disk where the directory should be located.
  """

    # Catch an exception when trying to delete the path because it may not yet exist.
    try:
        shutil.rmtree(path)
    except:
        pass

    os.makedirs(path)


def read_text(filename):
    """Read the content from a file

  Args:
    filename: The filename (path)
  """
    with open(filename, "r") as f:
        text = f.read()
    return text


def load_local_dataset(path):
    """Load a paired dataset (in png format)

  Args:
    path: The dataset root
  """
    image_files = sorted(glob.glob(os.path.join(path, "images", "*.jpg")))
    label_files = sorted(glob.glob(os.path.join(path, "labels", "*.jpg")))

    image_batch = []
    label_batch = []

    for image_file, label_file in zip(image_files, label_files):
        assert image_file.split("/")[-1] == label_file.split("/")[
            -1], "the dataset has unpaired data"
        image = np.array(PIL.Image.open(image_file).convert(mode="RGB"))
        label = np.array(PIL.Image.open(label_file).convert(mode="L"))
        image = np.array(image / 255.0 * 2.0 - 1.0, np.float32)
        label = np.expand_dims(np.array((label / 255.0 > 0.5), np.float32), axis=-1)
        image_batch.append(image)
        label_batch.append(label)
    return np.stack(image_batch), np.stack(label_batch)


def main(_):
    # Create the application.
    app = Application(app_filename = FLAGS.app_filename)

    # Startup the bridge to get data.
    node = app.find_node_by_name("ball_navigation_training_samples")
    bridge = packages.ml.SampleAccumulator(node)
    app.start()

    try:
        # Create an iterator over the data using tf.data.
        dataset = get_dataset(bridge)
        data_dict = dataset.make_one_shot_iterator().get_next()

        # Create the model.
        input_image = data_dict[COLOR_IMAGE_NAME]
        labels = data_dict[LABEL_IMAGE_NAME]
        unet_network, frozen_weights = create_unet()
        predictions, logits = unet(input_image, unet_network)

        # Create the model for validation.
        input_image_validation = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        labels_validation = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        predictions_validation, logits_validation = unet(input_image_validation, unet_network)

        # Create the loss function.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(labels, shape=[-1]), logits=tf.reshape(logits, shape=[-1]))
        # Use class weights since the ball is very small, and we want to account for this
        # imbalance. Otherwise, the model predicts only background.
        class_weights = tf.reshape(labels * (FLAGS.ball_class_weight - 1.0) + 1.0, [-1])
        loss = tf.reduce_mean(loss * class_weights)

        loss_validation = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(labels_validation, shape=[-1]),
            logits=tf.reshape(logits_validation, shape=[-1]))
        # Use class weights since the ball is very small, and we want to account for this
        # imbalance. Otherwise, the model predicts only background.
        class_weights = tf.reshape(labels_validation * (FLAGS.ball_class_weight - 1.0) + 1.0, [-1])
        loss_validation = tf.reduce_mean(loss_validation * class_weights)

        # Create the training op.
        var_list = tf.trainable_variables()
        var_list = list(filter(lambda x: x not in frozen_weights, var_list))

        # Create the training op.
        global_step = tf.placeholder(tf.int32)
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 100, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(loss, var_list=var_list)

        # Setup logging summaries and checkpoints..
        tf.summary.image('input_image', input_image, max_outputs=1)
        tf.summary.image('ground_truth', labels, max_outputs=1)
        tf.summary.image('predictions', predictions, max_outputs=1)
        tf.summary.scalar('loss', loss)
        tf.summary.image('valid_input_image', input_image_validation, max_outputs=4)
        tf.summary.image('valid_ground_truth', labels_validation, max_outputs=4)
        tf.summary.image('valid_predictions', predictions_validation, max_outputs=4)
        tf.summary.scalar('valid_loss', loss_validation)
        summaries = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
        saver = tf.train.Saver()

        # Setup logging folders.
        ckpt_dir = os.path.join(FLAGS.train_logdir, 'ckpts')
        log_dir = os.path.join(FLAGS.train_logdir, 'logs')
        ckpt_file = os.path.join(ckpt_dir, 'model')

        # Limit the GPU memory usage.
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_usage
        # config.gpu_options.allow_growth = True

        if not FLAGS.checkpoint:
            make_dir(ckpt_dir)
            make_dir(log_dir)

        # wait until we get enough samples
        while True:
            num = bridge.get_sample_count()
            if num >= kSampleNumbers:
                break
            time.sleep(1.0)
            print("waiting for samples samples: {}".format(num))
        print("Starting training")

        # Training loop.
        with tf.Session(config=config) as sess:

            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            tf.train.write_graph(sess.graph, ckpt_dir, 'graph.pb', as_text=False)

            if FLAGS.checkpoint:
                saver.restore(sess, os.path.join(ckpt_dir, FLAGS.checkpoint))
                print("Checkpoint Loaded - {}".format(FLAGS.checkpoint))

                # assumes the number of steps is at the end of the
                # checkpoint file (separated by a '-')
                init_step = int(FLAGS.checkpoint.split("-")[-1]) + 1
            else:
                init_step = 0

            image_batch, label_batch = load_local_dataset(FLAGS.validation_datadir)
            print("image:", np.min(image_batch), np.max(image_batch))
            print("label:", np.min(label_batch), np.max(label_batch))
            print("Start training from step: {}".format(init_step))
            for step in range(init_step, FLAGS.training_steps):
                if step % FLAGS.summary_every == 0:
                    [summaries_to_write, loss_value, loss_validation_value, _] = sess.run(
                        [summaries, loss, loss_validation, train_op],
                        feed_dict={
                            input_image_validation: image_batch,
                            labels_validation: label_batch
                        })
                else:
                    sess.run([train_op])
                train_writer.add_summary(summaries_to_write, step)
                tf.logging.info('step: {}, loss: {} (valid: {})'.format(step, loss_value,
                                                                        loss_validation_value))

                if step % FLAGS.save_every == 0 or step == FLAGS.training_steps - 1:
                    saver.save(sess, ckpt_file, global_step=step)

                    frozen_file = os.path.join("{}-{}-frozen.pb".format(ckpt_file, step))

                    freeze_graph.freeze_graph(
                        input_graph=os.path.join(ckpt_dir, 'graph.pb'),
                        input_saver="",
                        input_binary=True,
                        input_checkpoint=os.path.join("{}-{}".format(ckpt_file, step)),
                        output_node_names=OUTPUT_NAME,
                        restore_op_name="save/restore_all",
                        filename_tensor_name="save/Const:0",
                        output_graph=frozen_file,
                        clear_devices=True,
                        initializer_nodes="")
                    print("Saved frozen model at {}".format(frozen_file))
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt")
    except tf.errors.OutOfRangeError:
        print("Exiting due to training data stall")
    app.stop()


if __name__ == '__main__':
    tf.app.run()
