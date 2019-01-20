FROM tensorflow/tensorflow:1.12.0-gpu-py3

COPY dssm /root/code/dssm/
COPY requirements.txt /root/code/dssm

WORKDIR /root/code/dssm

