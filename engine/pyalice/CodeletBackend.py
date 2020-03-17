'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import time
from threading import Thread
from .CodeletFlowControl import CodeletFlowControl


class CodeletBackend(Thread):
    """Python CodeletBackend object that helps Python Codelet to communicate to the C++ PyCodelet

    Args:
        frontend (Codelet): the user defined codelet in python
        *args, **kwargs: threading.Thread arguments

    Attributes:
        frontend (Codelet): same as above
        flow_controller (CodeletFlowControl): the execution primitive that helps the backend to
          synchronize with C++ PyCodelet. The flow_controller will execute backend's callable
          attributes based on requests from C++ side.
  """

    def __init__(self, frontend_class, bridge, logger, node_name, *args, **kwargs):
        super(CodeletBackend, self).__init__(*args, **kwargs)
        self.frontend = frontend_class(self, logger, node_name)
        self.bridge = bridge
        self.__flow_controller = CodeletFlowControl(self, self.bridge)

    def __str__(self):
        return self.frontend._get_name()

    def __repr__(self):
        return __str__

    # backend functions for start, tick and stop
    def py_start(self):
        self.frontend.get_config_from_backend()
        self.frontend.start()
        self.frontend.set_config_to_backend()

    def py_tick(self):
        self.frontend.get_config_from_backend()
        self.frontend.tick()
        self.frontend.set_config_to_backend()

    def py_stop(self):
        self.frontend.get_config_from_backend()
        self.frontend.stop()
        self.frontend.set_config_to_backend()

    # overriding the threading.Thread run function
    def run(self):
        self.frontend.log_debug("Launched...".format(self))
        while self.__flow_controller.run():
            pass
        self.frontend.log_debug("Stopped...".format(self))
