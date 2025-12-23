FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

# =========================
# Base system
# =========================
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python-is-python3 \
    git build-essential wget curl ca-certificates \
    locales sudo \
    gnupg2 lsb-release bzip2 \
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# =========================
# ROS 2 repository + key
# =========================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN ARCH="$(dpkg --print-architecture)" && \
    echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
> /etc/apt/sources.list.d/ros2.list

# =========================
# ROS 2 Jazzy
# =========================
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete \
 && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update

# Source ROS for all shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# =========================
# Conda (Miniforge)
# =========================
WORKDIR /tmp

RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "arm64" ]; then \
      MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
      MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    fi && \
    curl -L -o Miniforge3.sh "${MINIFORGE_URL}" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm -f Miniforge3.sh && \
    /opt/conda/bin/conda clean -a -y

# Make conda available (without forcing a workspace cd)
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash

# =========================
# Python venv + packages (your original setup)
# =========================
ENV VENV_PATH=/opt/venv
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:/opt/conda/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    numpy scipy matplotlib tqdm \
    jupyterlab ipykernel \
    pandas \
    transformers datasets accelerate

# PyTorch: use CUDA wheels on amd64; fallback to default wheels on arm64
RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "${ARCH}" = "amd64" ]; then \
      pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 ; \
    else \
      pip install --no-cache-dir torch torchvision torchaudio ; \
    fi

WORKDIR /root
CMD ["bash"]
