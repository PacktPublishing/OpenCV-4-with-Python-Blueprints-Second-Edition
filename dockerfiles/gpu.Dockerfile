FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update && apt-get install -y \
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
        libgtk2.0-dev \
        # Optional
        libtbb2 libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev \
        libdc1394-22-dev \
        qt4-default \
        # Missing libraries for GTK and wxPython dependencies
        libatk-adaptor \
        libcanberra-gtk-module \
        x11-apps \
        libgtk-3-dev \
        # Tools
        imagemagick \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION="4.2.0"

WORKDIR /
RUN wget --output-document cv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && unzip cv.zip \
    && wget --output-document contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip contrib.zip \
    && mkdir /opencv-${OPENCV_VERSION}/cmake_binary

# Install numpy, since1. it's required for OpenCV
RUN pip install --upgrade pip && pip install --no-cache-dir numpy==1.18.1

RUN cd /opencv-${OPENCV_VERSION}/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
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
        -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        .. \
    && make install \
    && rm /cv.zip /contrib.zip \
    && rm -r /opencv-${OPENCV_VERSION} /opencv_contrib-${OPENCV_VERSION}

# RUN ln -s \
#   /usr/local/python/cv2/python-3.8/cv2.cpython-38m-x86_64-linux-gnu.so \
#   /usr/local/lib/python3.8/site-packages/cv2.so

RUN pip install --upgrade pip && pip install --no-cache-dir pathlib2 wxPython==4.0.5

RUN pip install --upgrade pip && pip install --no-cache-dir scipy==1.4.1 matplotlib==3.1.2 requests==2.22.0 ipython numba==0.48.0 jupyterlab==1.2.6 rawpy==0.14.0  # Rawpy is required for HDR & Panorama (processing .CR2 files)


CMD bash
