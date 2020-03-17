#!/bin/bash
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
DUMP_DIR=$(mktemp -d)
CRASH_BINARY=./engine/alice/tests/crashdump_binary
SYM_DIR=$DUMP_DIR/syms
ST_FILE=$DUMP_DIR/stacktrace.txt
SYM_FILENAME=crashdump_binary.sym

mkdir -p $DUMP_DIR && wait &&
echo $DUMP_DIR &&
$CRASH_BINARY --minidump_path=$DUMP_DIR && wait

# Checks dump file in $DUMP_DIR
DUMP_FILE=($DUMP_DIR/*)
echo $DUMP_FILE

# Checks number of files in $DUMP_DIR
NUM_DUMP=${#DUMP_FILE[@]}
if [[ ! $NUM_DUMP -eq 1 ]]; then
    echo "Only 1 dump file expected, yet has "$NUM_DUMP" instead"
    return 1
fi

# Extracts symbols
mkdir -p $SYM_DIR
./external/breakpad/dump_syms $CRASH_BINARY $SYM_DIR > $SYM_DIR/$SYM_FILENAME &2>/dev/null && wait

HASH=$(head -n1 $SYM_DIR/$SYM_FILENAME | awk '{print $4}')
mkdir -p $SYM_DIR/crashdump_binary/$HASH
mv $SYM_DIR/$SYM_FILENAME $SYM_DIR/crashdump_binary/$HASH/ && wait

# Extracts stacktrace
./external/breakpad/minidump_stackwalk $DUMP_FILE $SYM_DIR > $ST_FILE &2>/dev/null && wait
FOUND_LINES=(grep "crashdump_binary!main" $ST_FILE)

NUM_FOUND_LINES=${#FOUND_LINES[@]}
if [[ $NUM_FOUND_LINES -eq 0 ]]; then
    echo "Stacktrace file seems invalid"
    return 1
fi