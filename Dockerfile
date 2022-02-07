FROM nvcr.io/nvidia/pytorch:21.05-py3
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    python-setuptools swig emacs\
    python-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install tqdm tensorboard torchmetrics torch-fidelity ipywidgets urllib3 tensorboardX spacy numpy tensorflow
RUN pip3 install ujson transformers==4.2.2
