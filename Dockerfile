# the base image is a minimal installation of conda https://hub.docker.com/r/continuumio/miniconda3
# image = set of instructions
# container = virtual machine
FROM continuumio/miniconda3:latest

# Update
RUN apt-get update && apt-get upgrade -y

# Avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# By default, listen on port 5000
EXPOSE 5000/tcp

# Copy the dependencies file to the working directory
COPY requirements.txt .

# install  pip in conda base environment
RUN conda install pip -y

# Install all the dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY models ./models
COPY src ./src
COPY .env .
COPY app.py .

# run flask app using gunicorn webserver. this is the entry point. Runs the app.py file, so the models can be accessible
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]