# Extending image with tensorflow:
FROM tensorflow/tensorflow

RUN apt install python36 -y

ENV CUDA_VISIBLE_DEVICES="0"