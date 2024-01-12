FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update &&  \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mmengine==0.5.0  && \
    pip install --no-cache-dir mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html && \
    pip install --no-cache-dir mmdet==3.0.0rc5 wandb

COPY . /mmyolo
    #git clone https://github.com/jrchen2022a/mmyolo.git /mmyolo && \
    #        git checkout dev-custom && \
RUN cd /mmyolo && \
    pip install -r requirements/albu.txt && \
    pip install --no-cache-dir -e .
RUN pip uninstall yapf -y && \
    pip install yapf==0.32.0
RUN wandb offline
WORKDIR /mmyolo