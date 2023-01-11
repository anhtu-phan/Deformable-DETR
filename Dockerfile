FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV CUDA_HOME=/usr/local/cuda
RUN rm -rf /opt/pytorch

RUN pip uninstall -y torch torchvision torchtext Pillow

RUN pip --no-cache install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip --no-cache-dir install Cython wandb

COPY . /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /workspace