'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
import copy
import numpy as np
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import json
import shutil
import sys
import tensorflow as tf
import os

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_stream_data

from engine.pyalice import *
import apps.samples.yolo
import packages.ml


def _main():
    '''Entry point to the online training app.'''
    # Read config file.
    base_dir = "apps/samples/yolo/"
    config_path = os.path.join(base_dir, "keras-yolo3/configs/isaac_object_detection.json")
    config = {}
    with open(config_path) as f:
        config = json.load(f)

    # Read in training parameters (output path, learning rate, etc), classes names, and anchors.
    log_dir = os.path.expanduser(config['log_dir'])
    classes_path = config['classes_path']
    anchors_path = config['anchors_path']
    num_epochs_stage1 = config['num_epochs_stage1']
    num_epochs_total = config['num_epochs_total']
    gamma = config['gamma']
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # Create the bridge and start the app.
    app = Application(config["app_filename"])

    node = app.find_node_by_name("yolo_training.object_detection_training_samples")
    bridge = packages.ml.SampleAccumulator(node)
    app.start()

    try:
        # Limit the GPU memory usage to be 50%
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=tf_config))

        # Input size of the image to the yolo network, should be multiple of 32.
        input_shape = (416, 416)

        # Tell whether the model is tiny yolo or yolo and create the corresponding model.
        is_tiny_version = len(anchors) == 6
        if is_tiny_version:
            model = create_tiny_model(
                input_shape,
                anchors,
                num_classes,
                freeze_body=2,
                weights_path='../yolo_pretrained_models/yolo-tiny.h5')
        else:
            model = create_model(
                input_shape,
                anchors,
                num_classes,
                freeze_body=2,
                weights_path='../yolo_pretrained_models/yolo.h5')

        # Create the output folder if necessary.
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)

        # Setup learning schedule.
        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(
            os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=float(gamma), patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

        num_train = 5000    # assume there are 5000 images per epoch
        num_val = 200    # run 200 iterations for evaluation

        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            model.compile(
                optimizer=Adam(lr=float(config['lr_stage1'])),
                loss={
            # use custom yolo_loss Lambda layer.
                    'yolo_loss': lambda y_true, y_pred: y_pred
                })

            batch_size = int(config['batch_size_stage1'])
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, batch_size))
            model.fit_generator(
                data_generator(bridge, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator(bridge, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=num_epochs_stage1,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
            model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(
                optimizer=Adam(lr=float(config['lr_stage2'])),
                loss={
                    'yolo_loss': lambda y_true, y_pred: y_pred
                })    # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = int(config['batch_size_stage2']
                            )    # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, batch_size))
            model.fit_generator(
                data_generator(bridge, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator(bridge, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=num_epochs_total,
                initial_epoch=num_epochs_stage1,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))
    except KeyboardInterrupt:
        print("Exiting yolo training due keyboard interrupt")
    # Further training if needed.
    app.stop()


def read_text(filename):
    """Read the content from a file.

    Args:
      filename: the absolute path of the file.

    Returns:
      The content of the file.
    """
    with open(filename, "r") as f:
        text = f.read()
    return text


def get_classes(classes_path):
    """Read the classes name from file.

    Args:
      classes_path: the absolute path of the file which contains the name of categories.

    Returns:
      A list of names of all classes.
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """Loads the anchors from a file.

    Args:
      anchors_path: the absolute path of the file which contains all the anchors
                    used for the network.

    Returns:
      A numpy array contains all the anchors.
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape,
                 anchors,
                 num_classes,
                 load_pretrained=True,
                 freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    """Create the training model for Yolo v3.

    Args:
      input_shape: size of input image.
      anchors: anchors used for the model.
      num_classes: number of detection classes.
      load_pretrained: whether to use pretrained model.
      freeze_body: control the number of layers to be freezed.
      weights_path: path to the pretrained model.

    Returns:
      The loaded yolo model.
    """
    K.clear_session()    # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1, ),
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': 0.5
        })(model_body.output + y_true)

    model = Model([model_body.input] + y_true, model_loss)

    return model


def create_tiny_model(input_shape,
                      anchors,
                      num_classes,
                      load_pretrained=True,
                      freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    """Create the training model for tiny-yolo v3.

    Args:
      input_shape: size of input image.
      anchors: anchors used for the model.
      num_classes: number of detection classes.
      load_pretrained: whether to use pretrained model.
      freeze_body: control the number of layers to be freezed.
      weights_path: path to the pretrained model.

    Returns:
      The loaded tiny-yolo model.
    """
    K.clear_session()    # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1, ),
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': 0.7
        })(model_body.output + y_true)

    model = Model([model_body.input] + y_true, model_loss)

    return model


def data_generator(bridge, batch_size, input_shape, anchors, num_classes):
    """Data generator for online raining.

    Args:
      bridge: sim_bridge used to communicate with Isaac SDK.
      batch_size: batch size for training.
      input_shape: size of input image to the network.
      anchors: anchors used for the model.
      num_classes: number of detection classes.

    Returns:
      The data generator for training.
    """
    while True:
        image_data = []
        box_data = []
        if bridge.get_sample_count() < batch_size:
            continue
        sample = copy.deepcopy(bridge.acquire_samples(batch_size))
        if len(sample) < batch_size:
            continue
        for b in range(batch_size):
            image, box = get_stream_data(sample[b], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data] + y_true, np.zeros(batch_size)


if __name__ == '__main__':
    _main()
