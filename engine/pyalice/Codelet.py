'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import datetime
import json
from .CodeletHooks import TxHook, RxHook


class Codelet(object):
    """Python Codelet frontend object where the users will directly use the child class of the object
  and modifies start/tick/stop functions to suit their need

    Attributes:
        __name (str): the name of the codelet. This will be modified to contain node that it belongs
                      to.
        __backend (CodeletBackend): the backend object that should not be accessed by the user
        __logger (python logger object): the logger enables us to monitor python codelets
  """

    def __init__(self, backend, logger, node_name):
        self.__name = node_name + "/" + self.__class__.__name__
        self.__backend = backend
        self.__logger = logger
        self.__params = json.loads("{}")
        self.__params_delta = {}

    # convenient functions for logging
    def log_info(self, msg):
        self.__logger.info(msg, extra={"codeletname": self.__name})

    def log_warning(self, msg):
        self.__logger.warning(msg, extra={"codeletname": self.__name})

    def log_debug(self, msg):
        self.__logger.debug(msg, extra={"codeletname": self.__name})

    def log_error(self, msg):
        self.__logger.error(msg, extra={"codeletname": self.__name})

    def log_critical(self, msg):
        self.__logger.critical(msg, extra={"codeletname": self.__name})

    def log_exception(self, msg):
        self.__logger.exception(msg, extra={"codeletname": self.__name})

    """ Functions below provides direct access to the user
    """

    # add proto message hooks
    def isaac_proto_rx(self, proto_type, tag):
        assert self.__backend is not None, \
            "Fatal: backend has not been initialized ({})".format(self.__name)
        hook = RxHook(proto_type, tag, self.__backend.bridge)
        self.__backend.bridge.add_rx_hook(hook.tag)
        return hook

    def isaac_proto_tx(self, proto_type, tag):
        assert self.__backend is not None, \
            "Fatal: backend has not been initialized ({})".format(self.__name)
        hook = TxHook(proto_type, tag, self.__backend.bridge)
        self.__backend.bridge.add_tx_hook(hook.tag)
        return hook

    # wrapper functions for ticking behaviours configuration in isaac codelet
    def tick_on_message(self, rx):
        self.__backend.bridge.tick_on_message(rx.tag)

    def tick_blocking(self):
        self.__backend.bridge.tick_blocking()

    def tick_periodically(self, interval):
        self.__backend.bridge.tick_periodically(interval)

    def synchronize(self, *args):
        for rx1, rx2 in zip(args[:-1], args[1:]):
            assert isinstance(rx1, RxHook) and isinstance(rx2, RxHook), \
                "can not synchronize transmitting hook"
            self.__backend.bridge.synchronize(rx1.tag, rx2.tag)

    # wrapper functions for utility functions in isaac codelet
    def get_tick_time(self):
        return self.__backend.bridge.get_tick_time()

    def get_tick_dt(self):
        return self.__backend.bridge.get_tick_dt()

    def is_first_tick(self):
        return self.__backend.bridge.is_first_tick()

    def get_tick_count(self):
        return self.__backend.bridge.get_tick_count()

    # set/get isaac parameters
    def get_isaac_param(self, param_name):
        return self.__params[param_name]

    def set_isaac_param(self, param_name, param_value):
        self.__params[param_name] = param_value
        self.__params_delta[param_name] = param_value

    # allow publishing json dicts or messages to sight
    def __show(self, json_dict):
        if not isinstance(json_dict, dict):
            raise ValueError('Invalid datatype received : Expected a dictionary')
        if 'name' not in json_dict or 'type' not in json_dict:
            raise ValueError('Invalid Sight Json : Missing keys')
        json_str = json.dumps(json_dict)
        self.__backend.bridge.show(json_str)

    # publish variables (names and values) to sight
    def show(self, name, value, time=None):
        json_dict = {}
        json_dict["name"] = name
        json_dict["v"] = value
        json_dict["type"] = "plot"
        if time is not None:
            json_dict["t"] = time
        else:
            json_dict["t"] = self.get_tick_time() * 1000000
        self.__show(json_dict)

    def __getitem__(self, key):
        return self.get_isaac_param(key)

    def __setitem__(self, key, item):
        self.set_isaac_param(key, item)

    def get_config_from_backend(self):
        self.__params = json.loads(self.__backend.bridge.get_config())

    def set_config_to_backend(self):
        latest = json.loads(self.__backend.bridge.get_config())
        for key in self.__params_delta:
            latest[key] = self.__params_delta[key]
        self.__backend.bridge.set_config(json.dumps(latest))

    """ Functions below are to be overrided by the user
    """

    def start(self):
        pass

    def tick(self):
        pass

    def stop(self):
        pass
