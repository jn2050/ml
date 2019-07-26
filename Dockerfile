FROM ubuntu:18.04

RUN apt-get update &&\
    apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:apt-fast/stable &&\
    add-apt-repository -y ppa:graphics-drivers/ppa &&\
    apt-get install -y sudo bash &&\
    apt-get install -y apt-fast &&\
    apt-fast -y upgrade &&\
    ln -fs /usr/share/zoneinfo/Europe/Lisbon /etc/localtime &&\
    export DEBIAN_FRONTEND=noninteractive &&\
    apt-get install -y tzdata &&\
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-fast install -y ubuntu-drivers-common libvorbis-dev libflac-dev libsndfile-dev cmake \
         build-essential libgflags-dev libgoogle-glog-dev libgtest-dev google-mock zlib1g-dev \
         libeigen3-dev libboost-all-dev libasound2-dev libogg-dev libtool libfftw3-dev libbz2-dev \
         liblzma-dev libgoogle-glog0v5 gcc-6 gfortran-6 g++-6 doxygen graphviz libsox-fmt-all \
         parallel exuberant-ctags vim-nox python3-pip &&\
    apt-fast install -y tigervnc-standalone-server firefox lsyncd mesa-common-dev ack &&\
    apt-get install -y git-core wget

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 40 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-6 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-6 &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 40 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-7

RUN echo "mluser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers &&\
    groupadd -g 999 mluser &&\
    useradd -r -u 999 -g mluser mluser &&\
    mkdir -p /users/mluser &&\
    usermod -d /users/mluser mluser &&\
    usermod -s /bin/bash mluser &&\
    chown mluser:mluser /users/mluser

USER mluser
ENV HOME=/users/mluser
COPY dotfiles/ /users/mluser/dotfiles/
RUN sudo chown mluser $HOME/dotfiles
WORKDIR /users/mluser
RUN cp dotfiles/.bashrc /users/mluser &&\
    mkdir $HOME/.ssh && cp $HOME/dotfiles/ssh_config $HOME/.ssh/config &&\
    mkdir $HOME/.jupyter && cp $HOME/dotfiles/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

RUN mkdir downloads && cd downloads &&\
    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh &&\
    bash Anaconda3-2019.03-Linux-x86_64.sh -b
    
ENV PATH="$HOME/anaconda3/bin:$PATH"

RUN conda install -y -c conda-forge jupyter_contrib_nbextensions matplotlib feather-format uvicorn starlette awscli fire &&\
    conda install -y -c anaconda pip sqlalchemy pymysql bcrypt &&\
    pip install awscli --upgrade --user &&\
    pip install starlette-jwt &&\
    jupyter contrib nbextension install --user

RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda install -y -c pytorch -c fastai fastai

COPY nn/ /users/mluser/lib/nn
ENV PYTHONPATH="$HOME/lib/nn"

EXPOSE 8888
