FROM nvcr.io/nvidia/isaac-sim:6.1.0

USER root

ARG ISAACLAB_REF=v3.0.0-EA

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV ISAACLAB_PATH=/opt/IsaacLab
ENV ROS_DISTRO=jazzy

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]


# Base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    nano \
    vim \
    tmux \
    tree \
    htop \
    openssh-client \
    gnupg2 \
    lsb-release \
    locales \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install && locale-gen en_US.UTF-8


# ROS 2 Jazzy
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y universe \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu/ \
    $(. /etc/os-release && echo ${VERSION_CODENAME}) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update \
    -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy


# Miniforge / Conda
WORKDIR /tmp

RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "arm64" ]; then \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    fi && \
    curl -L "${MINIFORGE_URL}" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy

RUN /opt/conda/bin/conda config --system --set auto_activate_base false


# Isaac Lab 3.0 EA
WORKDIR /opt

RUN git clone \
    --branch ${ISAACLAB_REF} \
    --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git \
    /opt/IsaacLab

RUN ln -sfn /isaac-sim /opt/IsaacLab/_isaac_sim

WORKDIR /opt/IsaacLab


# Python package mirrors
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=12
ENV PIP_DISABLE_PIP_VERSION_CHECK=1


# Isaac Lab core
RUN /isaac-sim/python.sh --version

RUN ./isaaclab.sh -i core


# BFM / dataset dependencies
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    "zarr>=2.18,<3" \
    numcodecs \
    h5py \
    pandas \
    scipy \
    matplotlib \
    tqdm \
    tensorboard \
    pillow \
    opencv-python-headless \
    transformers \
    datasets \
    accelerate


# Shell setup
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'export ISAACLAB_PATH=/opt/IsaacLab' >> /etc/bash.bashrc && \
    echo "alias isaacpython='/isaac-sim/python.sh'" >> /etc/bash.bashrc && \
    echo "alias isaaclab='/opt/IsaacLab/isaaclab.sh'" >> /etc/bash.bashrc


# Workspace
RUN mkdir -p /workspace

WORKDIR /workspace

ENTRYPOINT []

CMD ["/bin/bash"]