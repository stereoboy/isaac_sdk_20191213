'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import time
import math

import engine.pyalice.tests
from engine.pyalice import *


def create_capnp_SO2_from_angle(angle, rotation_msg):
    angle = math.radians(angle)
    rotation_msg.q.x = math.cos(angle)
    rotation_msg.q.y = math.sin(angle)


def get_angle_from_capnp_SO2(rotation_msg):
    return math.degrees(math.atan2(rotation_msg.q.y, rotation_msg.q.x))


class MyPyCodeletProducer(Codelet):
    def start(self):
        self.tx = self.isaac_proto_tx("Pose2dProto", "pose")
        self.tick_periodically(0.1)
        self.cnt = 0
        self.set_isaac_param("x", 3.0)
        self["y"] = -2.1

    def tick(self):
        # get isaac parameters. Both ways are equivalent
        message = self.tx.init_proto()
        create_capnp_SO2_from_angle(60.0, message.rotation)
        message.translation.x = self["x"]
        message.translation.y = self.get_isaac_param("y")
        self.tx.publish()

        # test logger and tick utils
        tick_time = self.get_tick_time()
        tick_dt = self.get_tick_dt()
        is_first_tick = self.is_first_tick()
        get_tick_count = self.get_tick_count()

        desired_tick_time = self.cnt * 0.1
        desired_tick_dt = 0.1 if self.cnt != 0 else 0.0
        desired_is_first_tick = True if self.cnt == 0 else False
        desired_get_tick_count = self.cnt + 1

        self.log_info("pubtime: {}, acqtime: {}, uuid: {}".format(self.tx.pubtime(),
                                                                  self.tx.acqtime(),
                                                                  self.tx.message_uuid()))
        self.log_info("tick time: {} (dt = {}), first tick: {}, ticks #{}".format( \
          tick_time, tick_dt, is_first_tick, get_tick_count))
        epsilon = 0.1
        assert abs(tick_time - desired_tick_time) < epsilon, "{}".format(tick_time -
                                                                         desired_tick_time)
        assert abs(tick_dt - desired_tick_dt) < epsilon, "{}".format(tick_dt - desired_tick_dt)
        assert is_first_tick == desired_is_first_tick, "is_first_tick() fails to produce correct result"
        assert get_tick_count == desired_get_tick_count, "get_tick_count() fails"
        self.cnt += 1


class MyPyCodeletConsumer(Codelet):
    def start(self):
        self.rx = self.isaac_proto_rx("Pose2dProto", "pose")
        self.tick_on_message(self.rx)
        self.cnt = 0

    def tick(self):
        self.cnt += 1
        message = self.rx.get_proto()
        angle = get_angle_from_capnp_SO2(message.rotation)
        self.log_info("pubtime: {}, acqtime: {}, uuid: {}".format(self.rx.pubtime(),
                                                                  self.rx.acqtime(),
                                                                  self.rx.message_uuid()))
        assert abs(angle - 60.0) < 1e-14, \
          "angle not matched; expected: {:.15f}, actual: {:.6f}".format(60, angle)
        assert abs(message.translation.x - 3.0) < 1e-14, \
          "translation(x) not matched; expected: {:.6f}, actual: {:.6f}".format(3.0,
            message.translation.x)
        assert abs(message.translation.y + 2.1) < 1e-14, \
          "translation(x) not matched; expected: {:.6f}, actual: {:.6f}".format(-2.1,
            message.translation.y)

    def stop(self):
        assert self.cnt > 10, \
              "messaging system is unexpectedly slow (target: {}, actual: {})".format(10, self.cnt)


class MyPyCodeletProducerDouble(Codelet):
    def start(self):
        self.tx = self.isaac_proto_tx("Pose2dProto", "pose")
        self.tx2 = self.isaac_proto_tx("Pose2dProto", "pose2")
        self.tick_periodically(0.1)

    def tick(self):
        message1 = self.tx.init_proto()
        create_capnp_SO2_from_angle(60.0, message1.rotation)
        message1.translation.x = 3.0
        message1.translation.y = -2.1
        message2 = self.tx2.init_proto()
        create_capnp_SO2_from_angle(60.0, message2.rotation)
        message2.translation.x = 3.0
        message2.translation.y = -2.1
        message1_info = self.tx.publish()
        message2_info = self.tx2.publish(acqtime=message1_info["acqtime"])
        assert self.tx2.acqtime() == message1_info["acqtime"], "Invalid acqtime"
        assert self.tx2.pubtime() == message2_info["pubtime"], "Invalid pubtime"
        assert self.tx2.message_uuid() == message2_info["uuid"], "Invalid message uuid"
        self.log_info("tx - pubtime: {}, acqtime: {}, uuid: {}".format(
            self.tx.pubtime(), self.tx.acqtime(), self.tx.message_uuid()))
        self.log_info("tx2 - pubtime: {}, acqtime: {}, uuid: {}".format(
            self.tx2.pubtime(), self.tx2.acqtime(), self.tx2.message_uuid()))


class MyPyCodeletConsumerDouble(Codelet):
    def start(self):
        self.rx = self.isaac_proto_rx("UuidProto", "pose")
        self.rx2 = self.isaac_proto_rx("UuidProto", "pose2")
        self.tick_on_message(self.rx)
        self.tick_on_message(self.rx2)
        self.synchronize(self.rx, self.rx2)

    def tick(self):
        assert self.rx.acqtime() == self.rx2.acqtime()
        self.log_info("rx - pubtime: {}, acqtime: {}, uuid: {}".format(
            self.rx.pubtime(), self.rx.acqtime(), self.rx.message_uuid()))
        self.log_info("rx2 - pubtime: {}, acqtime: {}, uuid: {}".format(
            self.rx2.pubtime(), self.rx2.acqtime(), self.rx2.message_uuid()))


class PyCodeletImageReader(Codelet):
    def start(self):
        self.rx = self.isaac_proto_rx("ColorCameraProto", "rgb_image")
        self.tick_on_message(self.rx)
        self.cnt = 0

    def tick(self):
        self.cnt += 1
        img_msg = self.rx.get_proto()
        assert img_msg.image.rows == 4096, "Incorrect image rows {}".format(img_msg.image.rows)
        assert img_msg.image.cols == 4096, "Incorrect image cols {}".format(img_msg.image.cols)
        buffer_data = self.rx.get_buffer_content(img_msg.image.dataBufferIndex)
        assert len(buffer_data) == 4096 * 4096 * 3, "Incorrect image bytes length {}".format(
            len(buffer_data))

    def stop(self):
        assert self.cnt > 0, "ticking count {}".format(self.cnt)


def main():
    cases = ["py2py", "cpp2py", "py2cpp", "sync"]
    node_pycodelet_maps = [{
        "py_producer": MyPyCodeletProducer,
        "py_consumer": MyPyCodeletConsumer
    }, {
        "py_consumer": MyPyCodeletConsumer
    }, {
        "py_producer": MyPyCodeletProducer
    }, {
        "py_producer": MyPyCodeletProducerDouble,
        "py_consumer": MyPyCodeletConsumerDouble
    }]
    file_prefix = "engine/pyalice/tests/pycodelet_test_"

    for case, node_pycodelet_map in zip(cases, node_pycodelet_maps):
        app = Application(file_prefix + case + ".app.json")
        app.register(node_pycodelet_map)
        """ register all the python codelets and get an app handle that can start/stop the cpp and
        python thread together
        """
        app.start_wait_stop(2.0)

    # passing image buffer with message
    app = Application("engine/pyalice/tests/pycodelet_test_img.app.json")
    app.register({"py_reader": PyCodeletImageReader})
    app.start_wait_stop(2.0)


if __name__ == '__main__':
    main()
