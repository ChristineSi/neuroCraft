# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
#FROM python:3.10
FROM python:3.8.10

#      Slim version
#FROM python:3.10-slim

#      Tensorflow version
FROM tensorflow/tensorflow:2.13.0

#      Or tensorflow to run on Apple Silicon (M1 / M2)
# FROM armswdev/tensorflow-arm-neoverse:r23.08-tf-2.13.0-eigen

WORKDIR /app
# Copy everything we need into the image
COPY neurocraft neurocraft
#COPY api api
#COPY scripts scripts
#COPY requirements.txt requirements_docker.txt
COPY requirements_dev.txt requirements.txt
COPY setup.py setup.py
#COPY credentials.json credentials.json

# Install everything
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt requirements_docker.txt
RUN pip install -r requirements.txt
RUN pip install .

# Make directories that we need, but that are not included in the COPY
#RUN mkdir raw_data
#RUN mkdir models

# Install Uvicorn
RUN pip install uvicorn
RUN pip install fastapi

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD uvicorn neurocraft.api.fast:app --host 0.0.0.0 --port $PORT
