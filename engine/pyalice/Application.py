'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import datetime
import logging
import time
import traceback

from engine.pyalice import PybindApplication
from engine.pyalice import PybindPyCodelet
from engine.pyalice.CodeletBackend import CodeletBackend


class Application(object):
    """
    Central application similar to the C++ alice::Application
    """

    def __init__(self, app_filename, more_jsons=""):
        """
        Creates an Isaac application

        Args:
            app_filename: the main application json filename
            more_jsons: a comma-separated string of additional jsons to load for the app
        """
        self.app = PybindApplication(app_filename, more_jsons)

        # TODO - use ISAAC SDK logger
        FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('pycodelet')
        self.logger.setLevel(logging.DEBUG)
        self.default_logger_config = {"codeletname": "main"}

        # compile all pycodelet backends
        self.pycodelet_backends = {}
        self.pycodelet_frontends = {}

    def register(self, node_pycodelet_map):
        ''' Registers a python codelet '''
        for name, pycodelet_class in node_pycodelet_map.items():
            self.logger.debug("Launching {}".format(name), extra=self.default_logger_config)
            node = self.app.find_node_by_name(name)
            assert node.is_valid(), "Node not found in Isaac graph"
            bridge = PybindPyCodelet(node)
            backend = CodeletBackend(pycodelet_class, bridge, self.logger, name)
            self.pycodelet_backends[name] = backend
            self.pycodelet_frontends[name] = backend.frontend

    def start(self):
        ''' Starts the application '''
        self.logger.debug("Launching isaac core", extra=self.default_logger_config)
        self.app.start()
        self.logger.debug("Launching pycodelet threads", extra=self.default_logger_config)
        for _, backend in self.pycodelet_backends.items():
            backend.start()

    def stop(self):
        ''' Stops the application '''
        self.app.stop()
        for _, backend in self.pycodelet_backends.items():
            backend.join()
        self.logger.debug("Python Codelets All stopped...", extra=self.default_logger_config)

    def start_wait_stop(self, duration=None):
        '''
        Starts the application waits for the given duration and stops the application. If duration
        is not given the application will run forever or until Ctrl+C is pressed in the console.
        '''
        self.start()
        try:
            if duration is None:
                while True:
                    time.sleep(1.0)
            else:
                time.sleep(duration)
        except:
            traceback.print_exc()
        self.stop()

    def uuid(self):
        ''' Returns the UUID of the application as a stringl '''
        return self.app.uuid()

    def find_node_by_name(self, name):
        ''' Finds the node with the given name in the application '''
        return self.app.find_node_by_name(name)
