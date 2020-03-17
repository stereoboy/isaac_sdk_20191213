'''
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import engine.pyalice.tests
from engine.pyalice import *


def main():
    app_file = 'engine/pyalice/tests/pynode_test_status.app.json'
    node_name = 'status'
    invalid_node_name = 'dummy'

    app = Application(app_file)
    app.start_wait_stop(0.2)
    # test invalid node
    invalid_node = app.find_node_by_name(invalid_node_name)
    assert (not invalid_node.is_valid())
    assert (invalid_node.get_status() == bindings.Status.Invalid)
    # test node valid and running
    node = app.find_node_by_name(node_name)
    assert node.is_valid()
    assert node.get_status() == bindings.Status.Running

    # test node failure and application's more_jsons input with a single extra json
    more_jsons = 'engine/pyalice/tests/pynode_failure.json'
    app = Application(app_file, more_jsons)
    app.start_wait_stop(0.2)
    node = app.find_node_by_name(node_name)
    assert node.is_valid()
    assert node.get_status() == bindings.Status.Failure

    # test node success and application's more_jsons input with a single extra json
    more_jsons = 'engine/pyalice/tests/pynode_success.json'
    app = Application(app_file, more_jsons)
    app.start_wait_stop(0.2)
    node = app.find_node_by_name(node_name)
    assert node.is_valid()
    assert node.get_status() == bindings.Status.Success

    # test node failure and application's more_jsons input with multiple comma-separated extra jsons
    more_jsons = 'engine/pyalice/tests/pynode_success.json,engine/pyalice/tests/pynode_failure.json'
    app = Application(app_file, more_jsons)
    app.start_wait_stop(0.2)
    node = app.find_node_by_name(node_name)
    assert node.is_valid()
    assert node.get_status() == bindings.Status.Failure

    # test node success and application's more_jsons input with multiple comma-separated extra jsons
    more_jsons = 'engine/pyalice/tests/pynode_failure.json,engine/pyalice/tests/pynode_success.json'
    app = Application(app_file, more_jsons)
    app.start_wait_stop(0.2)
    node = app.find_node_by_name(node_name)
    assert node.is_valid()
    assert node.get_status() == bindings.Status.Success


if __name__ == '__main__':
    main()
