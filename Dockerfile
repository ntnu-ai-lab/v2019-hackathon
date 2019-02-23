# Extending image with tensorflow:
FROM tensorflow/tensorflow

RUN apt install python36 -y

# Decides what GPU you are using: 
# "0" For GPU:0
# "1" For GPU:1
ENV CUDA_VISIBLE_DEVICES="0"

