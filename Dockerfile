FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

MAINTAINER arne <arnedefauw@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    git && \
    apt-get clean
    
# Install miniconda to /miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y python=3.8 && \
conda install flask==1.1.2 && \
conda install pytorch==1.7.0 cudatoolkit=10.1 -c pytorch && \
conda clean --all

#Install Cython
RUN apt-get update
RUN apt-get -y install --reinstall build-essential
RUN apt-get -y install gcc
RUN pip install Cython

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

WORKDIR /work

#copy code:
COPY app.py /work
COPY utils.py /work
COPY bert_classifier/src/models.py /work/bert_classifier/src/

COPY models /work/models

# Arguments
ARG MODEL_DIR
ARG TYPESYSTEM_PATH

#copy model:
COPY $MODEL_DIR/*.pth /work/models/model.pth
COPY $MODEL_DIR/config.json /work/models/
COPY $MODEL_DIR/special_tokens_map.json /work/models/
COPY $MODEL_DIR/tokenizer_config.json /work/models/
COPY $MODEL_DIR/vocab.txt /work/models/

#copy typesystem:
COPY $TYPESYSTEM_PATH /work/typesystems/typesystem.xml

CMD python /work/app.py