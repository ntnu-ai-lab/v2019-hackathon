# Extending image with tensorflow:
FROM tensorflow/tensorflow:1.13.0rc2-gpu-py3

# Decides what GPU you are using: 
# "0" For GPU:0
# "1" For GPU:1
ENV CUDA_VISIBLE_DEVICES="0"

# Set working directory for container
WORKDIR /v2019-hackathon
COPY "./requirements.txt" "/v2019-hackathon/requirements.txt"

# Install git
RUN apt-get install git -y

# Install your python packages
RUN pip3 install --upgrade pip

# Add your dependencies in the requirements.txt
# To update the requirements with the dependencies in your virtual enviroment:
# $ pip freeze > requirements.txt
RUN pip3 install -r requirements.txt

COPY . /v2019-hackathon