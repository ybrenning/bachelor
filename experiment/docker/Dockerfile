FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

COPY requirements.txt requirements-optional.txt /code/

WORKDIR /code
RUN apt update && apt install git gcc g++ -y && \
    pip install -r requirements.txt && \
    pip install -r requirements-optional.txt && \
    # protobuf error: https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
    pip install 'protobuf<=3.20.1' --force-reinstall && \
    apt-get remove -y gcc g++
