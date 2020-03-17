#!/bin/bash
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# If a single parameter is given we interpret it as the map. If no parameter is given we use
# the default map.
MAP="apps/flatsim/demo_1"

while getopts :i:m: option
do
  case ${option} in
    m) MAP=${OPTARG};;
    i) MAP="apps/flatsim/demo_${OPTARG}";;
    *) break;
  esac
done
shift $((OPTIND -1))

engine/alice/tools/main --app apps/flatsim/flatsim.app.json --config "$MAP.config.json" --graph "$MAP.graph.json" $@
