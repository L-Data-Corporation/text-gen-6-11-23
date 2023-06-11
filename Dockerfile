FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL [ "/bin/bash","-c" ]
#Used for GPU setup
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

RUN apt update -y \
&& apt upgrade -y

RUN apt install wget -y \
&& apt install git -y \ 
&& apt install libaio-dev -y \
&& apt install libaio1 -y \
&& apt install curl -y \
&& apt-get install unzip \
&& apt-get install git-lfs

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& ./aws/install

RUN aws configure set aws_access_key_id "AKIA24XNHRWQFTQ7DF7V" \
&& aws configure set aws_secret_access_key "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" \
&& aws configure set region "us-east-1" \
&& aws configure set output "json"

RUN apt install python3.9 -y \
&& apt install python3-pip -y \
&& apt install python-is-python3 -y

RUN pip install --upgrade pip setuptools wheel

RUN pip install ninja

RUN pip install torch

RUN pip install datasets \ 
&& pip install transformers \ 
&& pip install accelerate \
&& pip install safetensors

RUN pip install sentencepiece

RUN pip install triton==1.0.0

RUN pip install git+https://github.com/microsoft/DeepSpeed.git@v0.8.2

RUN pip install git+https://github.com/microsoft/DeepSpeed-MII.git

RUN pip install wandb

RUN git lfs install

RUN pip install runpod

RUN pip install chromadb

RUN pip install sentence-transformers

ADD handler.py .

ADD memory.py .

ADD generate.py .

ADD start.sh /

RUN chmod +x /start.sh

CMD [ "/start.sh" ]

CMD [ "python", "-u", "/handler.py" ]