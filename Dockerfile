# Extending image with tensorflow:
FROM tensorflow/tensorflow:latest-gpu

RUN apt install python36 -y

# Decides what GPU you are using: 
# "0" For GPU:0
# "1" For GPU:1
ENV CUDA_VISIBLE_DEVICES="0"

# Set working directory for container
WORKDIR /v2019-hackathon


# Install git
RUN apt-get install git -y


# Install pip3 (parent image only comes with python2 stuff)
RUN apt-get install python3-pip -y

# Install your python packages
RUN pip3 install --upgrade pip

# Add your dependencies in the requirements.txt
# To update the requirements with the dependencies in your virtual enviroment:
# $ pip freeze > requirements.txt
RUN pip3 install -r requirements.txt
