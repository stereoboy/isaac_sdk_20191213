'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
''' Keras to Darknet
 This Script is used to convert Keras weights to darknet weights
 specifically for yolov3 architecture.
'''

import argparse
from keras.models import load_model
import numpy as np
import configparser
from collections import defaultdict
from io import StringIO


def unique_config_sections(config_file):
    """Module to parse the sections in the config file and assign
    unique names to each section

    Args:
        config_file: Path to yolov3 config file

    Returns: A StringIO object which contains the whole config

    """
    section_counters = defaultdict(int)
    output_stream = StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                section_counters[section] += 1
                _section = section + '_' + str(section_counters[section])
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def check_cfg(cfg_parser, num_classes):
    """Module to check if the config file matches the exact number of classes

    Args:
        cfg_parser: Configparser object which contains the whole config
        num_classes: Number of classes for detection

    Returns: Throws an error if class number mismatch is found

    """

    yolo_sections = []
    prev_conv_sections = []
    for idx, section_name in enumerate(cfg_parser.sections()):
        if section_name.startswith("yolo"):
            yolo_sections.append(section_name)
            prev_conv_sections.append(cfg_parser.sections()[idx - 1])
    classes_flags = [
        cfg_parser[section]["classes"] == str(num_classes) for section in yolo_sections
    ]
    filters_flags = [
        cfg_parser[section]["filters"] == str((num_classes + 5) * 3)
        for section in prev_conv_sections
    ]
    error_message = "Classes number mismatch, please change the classes in all [yolo] sections to {} " \
                    "and the filters in the [convolutional] sections before [yolo] sections to " \
                    "{}".format(num_classes, (num_classes + 5) * 3)
    assert (all(classes_flags) and all(filters_flags)), error_message


def keras_to_darknet(config_file, keras_weights_file, out_file, num_classes):
    """Module to convert Keras Yolo network weights to Darknet weights

    Args:
        config_file: Path to yolov3 config file
        keras_weights_file: Keras model weights file (.h5)
        out_file: Output path to save darknet weights (.weights)
        num_classes: Number of classes for detection

    """
    # Parsing the darknet config file and creating unique section names
    unique_config_file = unique_config_sections(config_file)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # Check if the config matches the class number
    check_cfg(cfg_parser, num_classes)

    model = load_model(keras_weights_file)

    fp = open(out_file, 'wb')

    # Attach the header at the top of the file
    # Darknet saves the version details in the first 20 bytes as a header.
    header = np.zeros(5).astype(np.int32)
    header.tofile(fp)

    # Keras Format - [conv_weights], [gamma, beta, mean, std]
    # Darknet Format - [beta, [gamma, mean, std], conv_weights]

    layers = model.layers

    # Dictionary of Batch Norm weights, Bias weights and Convolutional weights
    conv_weights = {}
    bias_weights = {}
    batch_norm_weights = {}

    for count in range(0, len(layers)):
        layer = layers[count]
        layer_type = layer.__class__.__name__

        # Skip layers other than convolution and batchnormalization
        if layer_type != "Conv2D" and layer_type != "BatchNormalization":
            continue
        # Accessing Batch normalization layer weights
        if layer_type == "BatchNormalization":
            batch_weights = layer.get_weights()
            gamma = np.array(batch_weights[0])
            beta = np.array(batch_weights[1])
            mean = np.array(batch_weights[2])
            std = np.array(batch_weights[3])
            bn_weights = []
            bn_weights.append(beta)
            bn_weights.append(gamma)
            bn_weights.append(mean)
            bn_weights.append(std)
            layer_name = int(str(layer.name)[20:])
            batch_norm_weights[layer_name] = bn_weights

        # Accessing Convolution layer weights
        else:
            conv = np.array((layer.get_weights()[0]))
            conv = np.transpose(conv, [3, 2, 0, 1])
            layer_name = int(str(layer.name)[7:])
            conv_weights[layer_name] = conv
            if np.array(layer.get_weights()).shape[0] == 2:
                bias = np.array((layer.get_weights()[1]))
                bias_weights[layer_name] = bias

        conv_weights_keys = conv_weights.keys()
        batch_norm_keys = batch_norm_weights.keys()
        bias_keys = bias_weights.keys()
        conv_weights_sorted = [conv_weights[x] for x in sorted(conv_weights_keys)]
        bn_weights_sorted = [batch_norm_weights[x] for x in sorted(batch_norm_keys)]
        bias_weights_sorted = [bias_weights[x] for x in sorted(bias_keys)]

    for section in cfg_parser.sections():
        if "convolutional" not in section:
            continue
        # Saving Batch Normalization weights
        if cfg_parser[section]['activation'] == 'linear':
            bn = np.array(bias_weights_sorted.pop(0))
        else:
            bn = np.array(bn_weights_sorted.pop(0))
        # Saving convolutional weights
        conv = np.array(conv_weights_sorted.pop(0))

        # Saving weights to file in Darknet format
        bn.astype(np.float32).tofile(fp)
        conv.astype(np.float32).tofile(fp)

    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras to darknet conversion.")
    parser.add_argument('-c', '--config_file', type=str, help="Config file path")
    parser.add_argument(
        '-i', '--keras_weights_file', type=str, help="Keras weights file (.h5) path")
    parser.add_argument('-o', '--out_file', type=str, help="Output file (.weights) path")
    parser.add_argument('-n', '--num_classes', type=int, help="Number of classes in detection")
    args = parser.parse_args()
    keras_to_darknet(args.config_file, args.keras_weights_file, args.out_file, args.num_classes)
