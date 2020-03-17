'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''
"""Segmentation model TensorFlow to TensorRT convertion script.

Converts a trained TensorFlow neural network which takes in an rgb image as input and
outputs semantic segmentation of a ball. The network architecture used is Unet.

Note: This script is specific to the frozen TensorFlow graph, produced by 'training.py' script
      from Ball Segmentation Tutorial.
"""

import unittest, PIL, sys, os
import tensorflow as tf
import tensorrt as trt
import uff                      # Note, this may require "sudo apt-get install uff-converter-tf"
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

# A path to the input model, see Machine Learning Workflow / Ball Segmentation tutorial or use one
# from: developer.nvidia.com/isaac/download/third_party/ball_segmentation_toy_model-20190626-tar-xz
FROZEN_GRAPH_FP = "model-9000-frozen.pb"

# A path to the test image. Please copy the test image into the current folder
# from: apps/samples/ball_segmentation/ball_validation_dataset/images/4724.jpg
TEST_IMAGE = "4724.jpg"

# Output path for trimmed TensorFlow graph (with ['OneShotIterator', 'IteratorGetNext', 'input']
# nodes replaced with a Placeholder type ['input'] node
TRIMMED_GRAPH_FP = "model-9000-trimmed.pb"

# Output path for device-agnostic TensorRT model
UFF_GRAPH_FP = "model-9000-trimmed.uff"

# Output path for device-specific TensorRT (CUDA) engine
PLAN_GRAPH_FP = "model-9000-trimmed.plan"

# Inference input shape, specified in the channels_last (or NHWC) order
# Note, this shape should match the size of the input image and TensorFlow model input shape
INPUT_SHAPE_NHWC = [1, 256, 512, 3]    # [images number][rows][columns][channels]

# Inference input shape, converted to channels_first (or NCHW) order
# Note, this shape should match the size of the input image and TensorRT model input shape
INPUT_SHAPE_NCHW = [
    INPUT_SHAPE_NHWC[0],                # [images number]   : 1
    INPUT_SHAPE_NHWC[3],                # [channels]        : 3
    INPUT_SHAPE_NHWC[1],                # [rows]            : 256
    INPUT_SHAPE_NHWC[2]                 # [columns]         : 512
]

def main():
    """Converts Tensorflow model-9000-frozen.pb into TensorRT model-9000-trimmed.uff"""

    if not os.path.exists(FROZEN_GRAPH_FP):
        print("Please make sure input files %s and %s exist in the current folder.\nPlease refer "
              "to Machine Learning Workflow / Ball Segmentation tutorial and this script`s source."
              % (FROZEN_GRAPH_FP, TEST_IMAGE) )
        sys.exit(1)

    # Open graph from "model-9000-frozen.pb"
    graph = tf.GraphDef()
    with gfile.FastGFile(FROZEN_GRAPH_FP, 'rb') as f:
        graph.ParseFromString(f.read())

    # Note, model-9000-frozen.pb graph contains spurious nodes. We are going to remove them and
    # replace with a placeholder input node.  The input shape will also be set to INPUT_SHAPE.

    # Create temp graph with a correct placeholder, name it 'input',
    with tf.Graph().as_default() as placeholder:
        input = tf.placeholder(tf.float32, name="input", shape=INPUT_SHAPE_NHWC)
        output = tf.identity(input, name="output")

    # Replace nodes ['OneShotIterator', 'IteratorGetNext', 'input']
    # with the placeholder 'input' node, which we've just created
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(placeholder.as_graph_def().node[:1])
    output_graph.node.extend(graph.node[3:])            # remove first 3 nodes (including 'input')
    with tf.gfile.GFile(TRIMMED_GRAPH_FP, 'w') as f:
        f.write(output_graph.SerializeToString())

    # Convert patched graph to UFF
    uff.from_tensorflow(TRIMMED_GRAPH_FP, ["output"], output_filename=UFF_GRAPH_FP, quiet=True)


# --------------- Test code, Run TensorRT and TensorFlow inference, check sanity
class InferenceTestCase(unittest.TestCase):
    def setUp(self):
        """Setup. Load an image from test data to use as test input to DNN"""

        # load test image into an array
        image = PIL.Image.open(TEST_IMAGE)
        image.load()

        # reshape that array into 4D tensor with dimensions: [image number][row][column][channel]
        images = np.asarray(image, dtype=np.float32).reshape(INPUT_SHAPE_NHWC)

        # normalize from [0..255] to [-1..1]
        self.images_rgb = images / 255.0 * 2.0 - 1.0

        # transpose into 4D tensor with dimensions: [image number][channel][row][column]
        self.images_rgb_channels_first = np.transpose(self.images_rgb, (0, 3, 1, 2)).copy()

    def test_tensorflow(self):
        """Run Tensorflow inference on self.images_rgb, validate the result"""
        tf.reset_default_graph()

        with tf.Graph().as_default() as graph:
            # Load frozen model to Tensorflow
            with gfile.FastGFile(TRIMMED_GRAPH_FP, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                g_in = tf.import_graph_def(graph_def)
                graph.finalize()

            # Create Tensorflow session, run inference with self.images_rgb as an input
            with tf.Session(
                    graph=graph,
                    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

                # Run Tensorflow inference
                input, output = graph.get_tensor_by_name(
                    "import/input:0"), graph.get_tensor_by_name("import/output:0")
                tf_results = sess.run(output, feed_dict={input: self.images_rgb})

                # Flatten the tensor and output inference results (raw floating point numbers)
                tf_result = np.array(tf_results[0]).flatten()
                print("Tensorflow: ", tf_result)

                # Convert tensor to RGB segmentation image and save to 4724.jpg.tensorflow.png
                image = PIL.Image.fromarray(tf_result.reshape(INPUT_SHAPE_NHWC[1:-1]) * 255.0)
                image.convert("RGB").save(TEST_IMAGE + ".tensorflow.png")


                # Check the results. (note, test is specific to model-9000-frozen.pb and 4724.jpg)
                self.assertAlmostEqual(tf_result[0], 0.025817014, places=6)
                self.assertAlmostEqual(tf_result[-1], 0.08415297, places=6)

    def test_tensor_rt(self):
        """Run TensorRT inference on self.images_rgb, validate the result"""
        import pycuda.driver as cuda    # sudo apt-get install python-pycuda OR pip3 install pycuda
        import pycuda.autoinit

        # Create Logger. Note: you can change severity to INFO or VERBOSE for debug information
        G_LOGGER = trt.Logger(trt.Logger.Severity.ERROR)

        # Create TensorRT engine (model-9000-trimmed.plan) from the model (model-9000-trimmed.uff)
        with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, trt.UffParser(
        ) as parser:
            # Set TensorRT engine parameters: batch size, workspace size, etc.  Please refer to
            # TensorRT manual and Isaac Machine Learning Workflow for more information
            builder.max_batch_size = 1
            builder.max_workspace_size = 2**32
            builder.fp16_mode = True

            # Identify indices of model inputs and outputs to bind to
            print("Registering input:", INPUT_SHAPE_NCHW[1:])
            parser.register_input("input", INPUT_SHAPE_NCHW[1:], trt.UffInputOrder.NHWC)
            parser.register_output("output")

            # Parse "model-9000-trimmed.uff" file to instantiate the model graph
            parser.parse(UFF_GRAPH_FP, network)
            print('Successfully parsed UFF model.')

            # Optimize model for target device, settings and CuDNN library version
            engine = builder.build_cuda_engine(network)
            print('Successfully built CUDA engine.')

            # Save optimized model to "model-9000-trimmed.plan"
            with open(PLAN_GRAPH_FP, "wb") as f:
                f.write(engine.serialize())

        # Allocate HOST memory for inference input and output
        h_input = cuda.pagelocked_empty(
            trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        h_output = cuda.pagelocked_empty(
            trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))

        # Allocate CUDA memory for inference input and output
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Run TensorRT Inference
        context = engine.create_execution_context()
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, self.images_rgb_channels_first, stream)
        context.execute_async(1, bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # Convert tensor to RGB segmentation image and save to 4724.jpg.tensorrt.png
        image = PIL.Image.fromarray(h_output.reshape(INPUT_SHAPE_NHWC[1:-1]) * 255.0)
        image.convert("RGB").save(TEST_IMAGE + ".tensorrt.png")

        # Output inference results (raw floating point numbers)
        print("TensorRT:   ", h_output)

        # Check the results. (note, test is specific to model-9000-frozen.pb and 4724.jpg)
        self.assertAlmostEqual(h_output[0], 0.025817007, places=6)
        self.assertAlmostEqual(h_output[-1], 0.08415297, places=6)


if __name__ == "__main__":
    main()
    unittest.main(exit=False)
