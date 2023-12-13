# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
#FROM python:3.10
FROM python:3.10.6-buster

#      Slim version
#FROM python:3.10-slim

#      Tensorflow version
# FROM tensorflow/tensorflow:2.13.0

#      Or tensorflow to run on Apple Silicon (M1 / M2)
# FROM armswdev/tensorflow-arm-neoverse:r23.08-tf-2.13.0-eigen

WORKDIR /prod
# Copy everything we need into the image

#COPY api api
#COPY scripts scripts
#COPY requirements.txt requirements_docker.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY neurocraft neurocraft
COPY setup.py setup.py
#COPY credentials.json credentials.json

ENV LD_PRELOAD=/usr/local/lib/python3.10/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0
# Install everything
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt requirements_docker.txt

RUN pip install .

COPY Makefile Makefile
RUN make reset_local_files

# RUN apt-get update && apt-get install -y libtbb2 libtbb-dev
# Make directories that we need, but that are not included in the COPY
#RUN mkdir raw_data
#RUN mkdir models

# # Install Uvicorn
# RUN pip install uvicorn
# RUN pip install fastapi

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD uvicorn neurocraft.api.fast:app --host 0.0.0.0 --port $PORT
