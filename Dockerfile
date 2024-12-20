FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Pre-install necessary dependencies
RUN pip install torch

# Pre-install packaging to avoid dependency errors
RUN pip install packaging

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Define the default command to run your application (if applicable)
CMD ["/bin/bash"]