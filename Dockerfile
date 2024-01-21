FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
MAINTAINER jrchen <jrchen3106@163.com>

ARG env="/mmyolo"
ENV WORK_PATH=$env

RUN apt-get update &&  \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install yapf==0.32.0 &&  \
    pip install mmengine==0.5.0  && \
    pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html && \
    pip install mmdet==3.0.0rc5 wandb

RUN pip install mmrazor==1.0.0

COPY . $WORK_PATH
#RUN git clone https://github.com/jrchen2022a/mmyolo.git "$WORK_PATH" && \
#    cd $WORK_PATH && \
#    git checkout dev-custom

RUN cd $WORK_PATH && \
    pip install -r requirements/albu.txt && \
    pip install --no-cache-dir -v .


WORKDIR $WORK_PATH