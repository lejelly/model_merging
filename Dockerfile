FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]

