FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

MAINTAINER arne <arnedefauw@gmail.com>

ARG MODEL_DIR
ARG TYPESYSTEM_PATH

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl && \
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

RUN pip install \
spacy==2.3.2 \
cloudpickle==1.6.0 \
torchtext==0.8.0 \
scikit-learn==0.23.2 \
transformers==3.4.0 \
scipy==1.4.1 \
numpy==1.18.5 \
tensorflow==2.3.1 \
keras==2.4.3 \
beautifulsoup4==4.9.3 \
fasttext==0.9.2 \
dkpro-cassis==0.5.0 \
pytest==6.1.1 \
ipython \
jupyter \
jupyterlab

#WORKDIR /work
#copy code:
#COPY app.py /work
#COPY code_bert_classifier/src/*.py /work/code_bert_classifier/src/
#copy model:
#COPY $MODEL_DIR/*.pth /work/models/model.pth
#COPY $MODEL_DIR/config.json /work/models/
#COPY $MODEL_DIR/special_tokens_map.json /work/models/
#COPY $MODEL_DIR/tokenizer_config.json /work/models/
#COPY $MODEL_DIR/vocab.txt /work/models/

#CMD python /work/app.py

EXPOSE 8888

VOLUME /notebook
WORKDIR /notebook

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip='0.0.0.0'"]
