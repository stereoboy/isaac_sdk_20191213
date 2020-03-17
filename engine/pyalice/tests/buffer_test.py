'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import PIL.Image
import time
import math
import numpy

from engine.pyalice import *


class PyImageProducer(Codelet):
    def start(self):
        self.tx = self.isaac_proto_tx("ColorCameraProto", "color")
        self.tick_periodically(0.1)
        self.log_info("pyimage start tick over")
        self.cnt = 0

        image_filename = self.get_isaac_param("color_filename")
        self.log_info("color_filename {}".format(image_filename))
        self.image = numpy.array(PIL.Image.open(image_filename).convert(mode="RGB"))

    def tick(self):
        # get isaac parameters. Both ways are equivalent
        message = self.tx.init_proto()
        message.colorSpace = 'rgb'

        focal = self.get_isaac_param("focal_length")
        message.pinhole.focal.x = focal[0]
        message.pinhole.focal.y = focal[1]

        center = self.get_isaac_param("optical_center")
        message.pinhole.center.x = center[0]
        message.pinhole.center.y = center[1]

        idx = self.tx.add_buffer(self.image)
        message.image.dataBufferIndex = idx - 1

        shape = self.image.shape
        message.image.rows = shape[0]
        message.image.cols = shape[1]
        message.image.channels = shape[2]

        message.image.elementType = 'uint8'
        self.tx.publish()

        self.cnt += 1

    def stop(self):
        assert self.cnt > 0, "ticking count {}".format(self.cnt)


class PyImageReader(Codelet):
    def start(self):
        self.rx1 = self.isaac_proto_rx("ColorCameraProto", "rgb_image_1")
        self.rx2 = self.isaac_proto_rx("ColorCameraProto", "rgb_image_2")
        self.tick_on_message(self.rx1)
        self.cnt = 0

    def tick(self):
        if not self.rx2.available():
            return
        img_msg1 = self.rx1.get_proto()
        img_msg2 = self.rx2.get_proto()

        assert img_msg1.image.rows == img_msg2.image.rows, "Incorrect image rows {} {}".format(
            img_msg1.image.rows, img_msg1.image.rows)
        assert img_msg1.image.cols == img_msg2.image.cols, "Incorrect image cols {} {}".format(
            img_msg1.image.cols, img_msg2.image.cols)
        assert img_msg1.image.channels == img_msg2.image.channels, "Incorrect image cols {} {}".format(
            img_msg1.image.channels, img_msg2.image.channels)
        assert img_msg1.image.elementType == img_msg2.image.elementType, "Incorrect element type {} {}".format(
            img_msg1.image.elementType, img_msg2.image.elementType)

        buffer_data_1 = self.rx1.get_buffer_content(img_msg1.image.dataBufferIndex)
        buffer_data_2 = self.rx1.get_buffer_content(img_msg1.image.dataBufferIndex)

        expected_bytes_length = img_msg1.image.rows * img_msg1.image.rows * img_msg1.image.channels
        assert len(buffer_data_1) == expected_bytes_length, "Incorrect buffer size {}".format(
            len(buffer_data_1))
        assert buffer_data_1 == buffer_data_2, "Inconsistent buffer data"
        self.cnt += 1

    def stop(self):
        assert self.cnt > 0, "ticking count {}".format(self.cnt)


def main():
    pycodelet_maps = {"py_image": PyImageProducer, "py_reader": PyImageReader}

    app = Application("engine/pyalice/tests/buffer_test.app.json")
    app.register(pycodelet_maps)
    """
    register all the python codelets and get an app handle that can start/stop the cpp and
    python thread together
    """
    app.start_wait_stop(2.0)


if __name__ == '__main__':
    main()
