#!/bin/bash
set -e
set -x

_python=python3.7
OPENCV_VERSION=4.1.2

sudo apt-get update && sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libgtk-3-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev \
        libdc1394-22-dev \
        qt4-default \
        libatk-adaptor \
        libcanberra-gtk-module \
        imagemagick

sudo ${_python} -m pip install numpy

wget --output-document cv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
  && unzip cv.zip \
  && wget --output-document contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
  && unzip contrib.zip \
  && mkdir opencv-${OPENCV_VERSION}/cmake_binary
cd opencv-${OPENCV_VERSION}/cmake_binary && cmake
  -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
  -D OPENCV_ENABLE_NONFREE=ON \
  -DCMAKE_INSTALL_PREFIX=$(${_python} -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which ${_python}) \
  -DPYTHON_INCLUDE_DIR=$(${_python} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(${_python} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  ..

make install
