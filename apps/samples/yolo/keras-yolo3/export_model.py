'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
import colorsys
import sys
import shutil
import os

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph

import onnxmltools

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


def load(model_path, anchors_path, class_path):
    """Load the trained model.

    Args:
      model_path: absolute path to the model.
      anchors_path: the absolute path of the file which contains all the anchors
                    used for the network.
      class_path:  the absolute path of the file which contains the name of categories.

    Returns:
      The loaded yolo/tiny-yolo model.
    """
    class_names = get_classes(class_path)
    anchors = get_anchors(anchors_path)
    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    # Load model, or construct model and load weights.
    num_anchors = len(anchors)
    num_classes = len(class_names)
    is_tiny_version = num_anchors==6 # default setting
    try:
        yolo_model = load_model(model_path, compile=False)
    except:
        yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
            if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        yolo_model.load_weights(model_path) # make sure model, anchors and classes match
    else:
        assert yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'
    input_image_shape = tf.constant([416, 416], shape=(2,))
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                       score_threshold=0.3, iou_threshold=0.45)
    print('{} model, anchors, and classes loaded.'.format(model_path))
    return yolo_model, is_tiny_version


def export_onnx(keras_model, output_path):
    """Export the model to the ONNX format.

    Args:
      keras_model: the loaded yolo/tiny-yolo model.
      output_path: output path of the ONNX model
    """
    onnx_model = onnxmltools.convert_keras(keras_model)
    onnxmltools.utils.save_model(onnx_model, output_path)


def export_pb(keras_model, output_folder, output_name):
    """Export the model to the tensorflow frozen model format.

    Args:
      keras_model: the loaded yolo/tiny-yolo model.
      output_path: output path of the pb model
    """
    K.set_learning_phase(0)
    pred_node_names = ["output_boxes", "output_scores", "output_classes"]
    print('output nodes names are: ', pred_node_names)
    # Get the current session
    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_folder, output_name, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', os.path.join(output_folder, output_name))


def export_models(model_path, anchors_path, classes_path, output_folder):
    """Export the model to both ONNX and pb files.

    Args:
      model_path: the absolute path to the trained h5 model.
      anchors_path: the absolute path of the file which contains all the anchors
                    used for the network.
      classes_path: the absolute path of the file which contains the name of categories.
      output_folder: output folder to export both pb and ONNX files.
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    yolo_model, is_tiny_version = load(model_path, anchors_path, classes_path)
    export_name = "tiny-yolo" if is_tiny_version else "yolo"
    export_onnx(yolo_model, os.path.join(output_folder, export_name + '.onnx'))
    export_pb(yolo_model, output_folder, export_name + '.pb')
    yolo_model.save(os.path.join(output_folder, export_name + '.h5'))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: $ python {0} [weights_path] [anchors_path] [classes_path] [output_folder]", sys.argv[0])
        exit()
    export_models(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
