FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get update &&\
    apt-get install -y --fix-missing \
        sudo bash wget curl rsync vim-nox uuid-dev gfortran-6 python python3-pip git git-core \
        ffmpeg libsm6 libxext6 iputils-ping postgresql-client

ENV DOCKER_VER=18.06.3-ce    
ENV DOCKER_URL=https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VER}.tgz
RUN curl -fsSLO $DOCKER_URL &&\
    tar xzvf docker-${DOCKER_VER}.tgz --strip 1 -C /usr/local/bin docker/docker &&\
    rm docker-${DOCKER_VER}.tgz

ENV HOME=/users/ml
RUN echo "ml ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers &&\
    groupadd -g 999 ml &&\
    useradd -r -u 999 -g ml ml &&\
    mkdir -p $HOME &&\
    usermod -d $HOME ml &&\
    usermod -s /bin/bash ml &&\
    chown ml:ml $HOME &&\
    mkdir $HOME/downloads && chown ml:ml $HOME/downloads && chmod 777 $HOME/downloads
USER ml
WORKDIR $HOME

COPY files/ $HOME/files/
RUN sudo chown -R ml:ml $HOME &&\
    cp $HOME/files/.bashrc $HOME &&\
    cp $HOME/files/.exrc $HOME &&\
    cp $HOME/files/.vimrc $HOME &&\
    mkdir $HOME/.ssh && cp $HOME/files/ssh_config $HOME/.ssh/config &&\
    mkdir $HOME/.jupyter && cp $HOME/files/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py &&\
    mkdir $HOME/scripts && cp $HOME/files/ju.sh $HOME/scripts && chmod 777 $HOME/scripts/* &&\
    mkdir $HOME/dev

ENV PATH="$HOME/anaconda3/bin:$PATH"
ENV PATH="/usr/local/anaconda3/bin:$PATH"
ENV ANACONDA_VER=Anaconda3-2020.07-Linux-x86_64.sh
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]

RUN pip3 install --upgrade pip
RUN cd $HOME/downloads &&\
    wget -q https://repo.anaconda.com/archive/$ANACONDA_VER &&\
    bash $ANACONDA_VER -b &&\
    rm $ANACONDA_VER &&\
    conda update -y -n base -c defaults conda &&\
    conda init bash
RUN conda env create -f $HOME/files/environment.yml &&\
    echo "conda activate ml" >> ~/.bashrc

RUN conda install -y tensorflow-gpu
# RUN conda install -y -c fastai -c pytorch -c anaconda fastai gh anaconda
RUN pip install opencv-python

RUN jupyter contrib nbextension install --user

RUN sudo apt update &&\
    sudo apt -y install nodejs &&\
    sudo apt -y install npm &&\
    sudo npm install -g ijavascript &&\
    ijsinstall

RUN conda install -y -c rapidsai -c nvidia -c numba -c conda-forge cudf=0.18 python=3.8

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skip_cache
RUN pip install dl2050utils
RUN pip install dl2050nn

# RUN pip install dl2050utils==1.0.12
# RUN pip install dl2050nn==1.0.39

# COPY --chown=ml:ml lib/nn2/ $HOME/lib/nn2
# RUN pip install -e $HOME/lib/nn2
