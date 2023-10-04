# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/
FROM nvcr.io/nvidia/pytorch:23.08-py3
ENV FORCE_CUDA="1"

# Create user and install sudo
RUN useradd -m robesafe -s /bin/bash -g root -G sudo -u 1000

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "robesafe:perception"| chpasswd
RUN usermod -aG sudo robesafe
RUN adduser robesafe sudo
RUN apt update
# RUN apt upgrade -y
RUN apt install sudo libgl1-mesa-glx mesa-utils libglapi-mesa libqt5gui5  -y

RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/robesafe/.local/bin"

RUN mkdir /.cache
COPY requirements.txt /.cache

USER robesafe

# Install python requirements
RUN pip install -r /.cache/requirements.txt

WORKDIR /home/robesafe/Datasets
WORKDIR /home/robesafe/workspace
ENTRYPOINT /bin/bash
