FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update
RUN apt-get install libgl1 -y
RUN apt-get -y install python3-pip
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y libgtk2.0-dev
# RUN ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11
# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

CMD [ "python3", "track.py"]