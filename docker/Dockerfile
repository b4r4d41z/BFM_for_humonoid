# ============================================================
# BFM development environment
#
# Isaac Sim 6.1
# Isaac Lab 3.0
# ROS 2 Jazzy
# Miniforge / Conda
# PyTorch (provided by Isaac Sim / Isaac Lab environment)
# Zarr / HDF5 and common ML tools
#
# The BFM repository itself is NOT copied into this image.
# ============================================================

FROM nvcr.io/nvidia/isaac-sim:6.1.0

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

SHELL ["/bin/bash", "-c"]


# ============================================================
# Base development tools
# ============================================================

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

RUN git lfs install

RUN locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


# ============================================================
# ROS 2 Jazzy
# ============================================================

RUN curl -sSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo ${VERSION_CODENAME}) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update || true

ENV ROS_DISTRO=jazzy


# ============================================================
# Miniforge / Conda
#
# Conda is available for auxiliary environments.
# Isaac Sim itself continues using /isaac-sim/python.sh.
# ============================================================

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

# Makes `conda activate ...` available without replacing Isaac Sim Python.
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc


# ============================================================
# Isaac Lab
#
# Isaac Lab 3.0 -> Isaac Sim 6.1
# ============================================================

WORKDIR /opt

RUN git clone \
    --branch v3.0.0-EA \
    --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git \
    /opt/IsaacLab

# Isaac Lab expects an Isaac Sim installation.
RUN ln -s /isaac-sim /opt/IsaacLab/_isaac_sim

WORKDIR /opt/IsaacLab

RUN ./isaaclab.sh -i


# ============================================================
# BFM / dataset dependencies
#
# Install them into Isaac Sim's Python because this is the
# Python environment that will run Isaac Lab + BFM together.
# ============================================================

RUN /isaac-sim/python.sh -m pip install --upgrade pip

RUN /isaac-sim/python.sh -m pip install \
    zarr \
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


# ============================================================
# Shell environment
# ============================================================

ENV ISAACLAB_PATH=/opt/IsaacLab
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV PYTHONUNBUFFERED=1

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'export ISAACLAB_PATH=/opt/IsaacLab' >> /etc/bash.bashrc && \
    echo "alias isaacpython='/isaac-sim/python.sh'" >> /etc/bash.bashrc && \
    echo "alias isaaclab='/opt/IsaacLab/isaaclab.sh'" >> /etc/bash.bashrc


# ============================================================
# User workspace
#
# Nothing is copied here during build.
# Mount your repositories here when starting a container.
# ============================================================

RUN mkdir -p /workspace

WORKDIR /workspace


# Clear Isaac Sim's inherited headless entrypoint.
ENTRYPOINT []

CMD ["/bin/bash"]