# syntax = docker/dockerfile:experimental
#
# This file can build images for cpu and gpu env. By default it builds image for CPU.
# Use following option to build image for cuda/GPU: --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Here is complete command for GPU/cuda -
# $ DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/


ARG BASE_IMAGE=ubuntu:18.04
ARG DEV_IMAGE=ubuntu:18.04

FROM ${DEV_IMAGE} AS base

# This is useful for set this env
ARG BASE_IMAGE=ubuntu:18.04
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list


FROM base AS compile-image
MAINTAINER zhoubingcheng@dockerhub.datagrand.com
ENV PYTHONUNBUFFERED TRUE
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y ca-certificates g++ \
    python3.8-dev python3.8-distutils python3.8-venv \
    curl git build-essential cmake gcc gdb rsync openssh-server unzip wget vim \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py
RUN echo "set encoding=utf-8" > ~/.vimrc

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3m python3m /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN python -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools \
    -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG CUDA_VERSION=""

RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    TORCH_VISION_VER=$(curl --silent --location https://pypi.org/pypi/torchvision/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            pip install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION torchvision==$TORCH_VISION_VER+$CUDA_VERSION -f https://download.pytorch.org/whl/torch_stable.html \
            -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
        # Install the binary with the latest CUDA version support
        else \
            pip install --no-cache-dir torch torchvision -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
        fi \
    # Install the CPU binary
    else \
        pip install --no-cache-dir torch==$TORCH_VER+cpu torchvision==$TORCH_VISION_VER+cpu -f https://download.pytorch.org/whl/torch_stable.html \
         -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
    fi
RUN pip install --no-cache-dir requests>=2.25 -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
WORKDIR /workspace
ENV WORK_DIR /workspace
RUN mkdir /usr/local/cpp_libs
RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") \
    && echo "CUDA_VERSION:${CUDA_VERSION},TORCH_VER:${TORCH_VER}" \
    && cd /usr/local/cpp_libs && wget --no-verbose https://download.pytorch.org/libtorch/${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2B${CUDA_VERSION}.zip \
    && unzip -qq libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}+${CUDA_VERSION}.zip \
    && rm -rf libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}+${CUDA_VERSION}.zip
RUN cd /root && wget --no-verbose https://boostorg.jfrog.io/artifactory/main/release/1.65.0/source/boost_1_65_0.tar.gz && \
    tar zxf boost_1_65_0.tar.gz && mv boost_1_65_0 boost && rm -rf boost_1_65_0.tar.gz
RUN cd /root/boost && ./bootstrap.sh --without-libraries=python,wave,program_options --prefix=/usr/local/cpp_libs/boost \
    && ./b2 -j 8  install \
    && cd .. && rm -rf /root/boost
COPY cpp_libs/opencv /usr/local/cpp_libs/opencv
COPY 3rdparty ${WORK_DIR}/3rdparty
COPY apps ${WORK_DIR}/apps
COPY cmake ${WORK_DIR}/cmake
COPY modules ${WORK_DIR}/modules
COPY samples ${WORK_DIR}/samples
COPY scripts ${WORK_DIR}/scripts
COPY test ${WORK_DIR}/test
COPY CMakeLists.txt ${WORK_DIR}/CMakeLists.txt

COPY data ${WORK_DIR}/data
RUN cd ${WORK_DIR} && mkdir build && cd build && cmake .. -DJSON_Install=ON -DCMAKE_INSTALL_PREFIX=/usr/local/cpp_libs/inference \
    && make -j 8 install && cd .. && rm -rf build


FROM compile-image as dev-image

ENV INCLUDE_PATH /usr/local/cpp_libs/opencv/include:/usr/local/cpp_libs/libtorch/include:/usr/local/cpp_libs/boost/include:/usr/local/cpp_libs/inference/include
ENV LD_LIBRARY_PATH /usr/local/cpp_libs/libtorch/lib:/usr/local/cpp_libs/opencv/lib:/usr/local/cpp_libs/boost/lib:/usr/local/cpp_libs/inference/lib:

RUN mkdir -p /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

 # SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22
ENV TZ Asia/Shanghai
WORKDIR /workspace
ENV WORK_DIR /workspace

CMD /usr/sbin/sshd -D



