'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

# import all the cap'n'proto messages so that the pycodelet can decode them in backend
from .CapnpMessages import get_capnp_proto_schemata
CAPNP_DICT = get_capnp_proto_schemata()


class RxHook(object):
    """Python message rx hook that mirrors the isaac message rx hook

    Args:
        proto_type (str): the name of the proto type. The script will find the proto in the pool
        tag (str): the channel/tag of the rx hook (same definition as isaac message rx hook)
        backend_receive (function): python function that calls pybind codelet to receive message
        backend_available (function): python function that calls pybind codelet to detect message

    Attributes:
        proto_schema (capnp struct object): the capnp object that is able to encode/decode message
        the rest of them are the same as above
  """

    def __init__(self, proto_type, tag, bridge):
        assert proto_type in CAPNP_DICT, \
          "proto message type \"{}\" not registered".format(proto_type)
        self.proto_schema = CAPNP_DICT[proto_type]
        self.tag = tag
        self.bridge__ = bridge

    def acqtime(self):
        return self.bridge__.get_rx_acqtime(self.tag)

    def pubtime(self):
        return self.bridge__.get_rx_pubtime(self.tag)

    def message_uuid(self):
        return self.bridge__.get_rx_uuid(self.tag)

    def available(self):
        return self.bridge__.available(self.tag)

    def get_proto(self):
        encoded_message = self.bridge__.receive(self.tag)
        return self.proto_schema.from_bytes(encoded_message)

    def get_buffer_content(self, index):
        return self.bridge__.get_rx_buffer_content(self.tag, index)


class TxHook(object):
    """Python message tx hook that mirrors the isaac message tx hook

    Args:
        proto_type (str): the name of the proto type. The script will find the proto in the pool
        tag (str): the channel/tag of the tx hook (same definition as isaac message tx hook)
        backend_publish (function): python function that calls pybind codelet to publish message

    Attributes:
        proto_schema (capnp struct object): the capnp object that is able to encode/decode message
        the rest of them are the same as above
  """

    def __init__(self, proto_type, tag, bridge):
        assert proto_type in CAPNP_DICT, \
          "proto message type \"{}\" not registered".format(proto_type)
        self.proto_schema = CAPNP_DICT[proto_type]
        self.tag = tag
        self.bridge__ = bridge
        self.acqtime_cache = None
        self.pubtime_cache = None
        self.uuid_cache = None
        self.message_cache = None

    def init_proto(self, *args, **kwargs):
        self.acqtime_cache = None
        self.pubtime_cache = None
        self.uuid_cache = None
        self.message_cache = self.proto_schema.new_message(*args, **kwargs)
        return self.message_cache

    def acqtime(self):
        return self.acqtime_cache

    def pubtime(self):
        return self.pubtime_cache

    def message_uuid(self):
        return self.uuid_cache

    def publish(self, proto_message=None, acqtime=None):
        if proto_message is None:
            proto_message = self.message_cache    # store the message
        assert proto_message is not None, "the proto message has not been properly initialize"
        message_info = self.bridge__.publish(proto_message.to_bytes(), self.tag, acqtime)
        self.acqtime_cache = message_info["acqtime"]
        self.pubtime_cache = message_info["pubtime"]
        self.uuid_cache = message_info["uuid"]
        return message_info

    def add_buffer(self, buffer):
        return self.bridge__.add_tx_buffer(self.tag, buffer)
