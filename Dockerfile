FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python-is-python3 \
    git build-essential wget curl ca-certificates \
    locales sudo \
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    numpy scipy matplotlib tqdm \
    jupyterlab ipykernel \
    pandas \
    transformers datasets accelerate

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

CMD ["bash"]
