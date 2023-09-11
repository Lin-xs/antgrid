FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget \
  libgtest-dev libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev llvm-dev git-lfs \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python=3.8 pip cmake

# RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision

RUN pip install --no-cache-dir  torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

RUN pip install nvidia-pytriton fire

RUN mkdir antgrid

COPY . antgrid/

RUN cd antgrid && pip install -r requirements.txt

EXPOSE 8000 8001 8002

CMD bash