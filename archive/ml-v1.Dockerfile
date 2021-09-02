FROM ubuntu:18.04
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    ln -fs /usr/share/zoneinfo/Europe/Lisbon /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --fix-missing \
        ack \
        bash \
        build-essential \
        cmake \
        doxygen \
        exuberant-ctags \
        firefox \
        g++-6 \
        gcc-6 \
        gfortran-6 \
        git-core \
        google-mock \
        graphviz \
        libasound2-dev \
        libboost-all-dev \
        libbz2-dev \
        libeigen3-dev \
        libfftw3-dev \
        libflac-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgoogle-glog0v5 \
        libgtest-dev \
        liblzma-dev \
        libogg-dev \
        libsndfile-dev \
        libsox-fmt-all \
        libtool \
        libvorbis-dev \
        lsyncd \
        mesa-common-dev \
        parallel \
        python3-pip \
        software-properties-common \
        sudo \
        tigervnc-standalone-server \
        ubuntu-drivers-common \
        vim-nox \
        wget \
        zlib1g-dev 

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 40 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-6 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-6 &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 40 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-7

ENV HOME=/users/mluser
RUN echo "mluser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers &&\
    groupadd -g 999 mluser &&\
    useradd -r -u 999 -g mluser mluser &&\
    mkdir -p $HOME &&\
    usermod -d $HOME mluser &&\
    usermod -s /bin/bash mluser &&\
    chown mluser:mluser $HOME
USER mluser
WORKDIR $HOME

ENV PATH="$HOME/anaconda3/bin:$PATH"
RUN mkdir $HOME/downloads && chown mluser $HOME/downloads && cd $HOME/downloads &&\
    wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh && \
    bash $HOME/downloads/Anaconda3-2019.07-Linux-x86_64.sh -b && \
    conda update -y -n base conda

COPY files/ $HOME/files/
COPY nn/ $HOME/lib/nn
RUN sudo chown mluser $HOME/files && sudo chown mluser $HOME/lib && sudo chown mluser $HOME/lib/nn
RUN cp $HOME/files/.bashrc $HOME &&\
    cp $HOME/files/.exrc $HOME &&\
    cp $HOME/files/.vimrc $HOME &&\
    cp $HOME/files/.inputrc $HOME &&\
    mkdir $HOME/.ssh && cp $HOME/files/ssh_config $HOME/.ssh/config &&\
    mkdir $HOME/.jupyter && cp $HOME/files/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

RUN conda env create -f $HOME/files/environment.yml
RUN conda install -y -c pytorch -c fastai fastai

RUN conda init bash
ENV PATH /usr/local/anaconda3/bin:$PATH
RUN cd $HOME/lib/ && sudo -H /users/mluser/anaconda3/bin/pip install -e nn

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda activate ml && jupyter contrib nbextension install --user
RUN echo "conda activate ml" >> ~/.bashrc
RUN mkdir $HOME/scripts && echo "conda activate ml && jupyter notebook" > $HOME/scripts/ju.sh
EXPOSE 8888

# RUN pip install awscli --upgrade --user

# RUN cd $HOME/downloads && git clone https://github.com/universal-ctags/ctags.git && \
#     cd ctags && ./autogen.sh && ./configure && \
#     make && sudo make install