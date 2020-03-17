#!/bin/sh
#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################

# Some input files might be created by more complicated rules and placed in the genfiles folder.
# Let's move them to the current tree. We are operating on a safe environment here.
# TODO Will this create problems when we are not in a sandbox?
# FIXME This will only work for opt mode.
rsync -av bazel-out/k8-opt/genfiles/ .
rm -rf bazel-out/k8-opt/genfiles/

SPHINXOPTS=
SOURCEDIR="."
BUILDDIR="_sphinx_build"

# The conf file needs to be at the root of the source folder
ln -s engine/build/sphinx/conf.py .

# The _static directory needs to be at the root of the source folder too
ln -s engine/build/sphinx/_static .

# Run the sphinx tool (needs to be installed previously)
sphinx-build -M html $SOURCEDIR $BUILDDIR $SPHINXOPTS

# Lets find the name (no path and no extension) of the desired output file
filename=$(basename -- "$1")
NAME=${filename%.tar.gz}

# Compress the output folder with various HTML, CSS and Javascript files into a tar archive with
# the desired name and copy it to the requested output location.
mv $BUILDDIR/html/ $NAME
tar -zcf $NAME.tar.gz $NAME
mv $NAME.tar.gz $1
