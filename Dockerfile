# Use the official Ubuntu as the base image
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Update the package repository and install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11

# Set the default Python version to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN apt-get install -y python3-pip
WORKDIR /app
# Run a command to verify the Python version
CMD ["/bin/bash"]
