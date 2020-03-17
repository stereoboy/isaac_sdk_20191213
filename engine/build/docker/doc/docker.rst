Using Docker
============

Isaac SDK development can be done in Docker container, allowing teams to use a standard environment,
and use that environment inside of non-Linux operating systems, such as Windows and Mac OS. This
section describes how to build and run an Isaac SDK Docker container.
step in troubleshooting problems.

Installing Dependencies
-----------------------

Off the shelf docker is required to build Isaac SDK applications. To install them, run

.. code-block:: bash

   engine/build/docker/install_docker.sh

NVIDIA docker is required. Please follow the instructions at
`Installation Guide <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_ for
installation.

.. _create_docker_image:

Creating Isaac SDK Development Image
------------------------------------

1. After installing dependencies, run following script to create the ``isaacbuild`` image for Isaac
   SDK development:

   .. code-block:: bash

      engine/build/docker/create_image.sh

2. Create a cache volume for faster builds with the following command:

   .. code-block:: bash

      docker volume create isacc-sdk-build-cache

3. Run the container with the following command:

   .. code-block:: bash

      docker run --mount source=isaac-sdk-build-cache,target=/root -v `pwd`:/src/workspace -w /src/workspace --runtime=nvidia -it isaacbuild:latest /bin/bash

4. Run the following command inside the container to build Isaac SDK:

   .. code-block:: bash

      bazel build ...
