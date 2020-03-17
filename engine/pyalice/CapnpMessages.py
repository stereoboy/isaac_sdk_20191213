'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import os
import capnp
import glob

PATH = "**/*.capnp"    # the path to find all ".capnp" files
CLASS_NAME = "_StructModule"    # the name that pycapnp uses for its pycapnp schema object


def get_capnp_proto_schemata():
    '''Load all the capnp'n'proto schemata in the project. The function will glob through all the
  files with "*.capnp" extension name. It will return a dictionary that maps from the name of each
  capnp proto schema to the actual pycapnp schema object that can read/create capnp'n'proto message.

  Returns:
      A dictionary that maps from the name of capnp proto schema to the pycapnp proto schema object
  '''
    capnp_dict = {}
    for capnp_f in glob.glob(PATH):    # loop through the codebase to find all
        module = capnp.load(capnp_f)    # the .capnp files
        for name, obj in module.__dict__.items():    # register the name of the proto message
            if obj.__class__.__name__ == CLASS_NAME:    # type
                assert name not in capnp_dict
                capnp_dict[name] = obj    # store the capnp struct
    return capnp_dict
