FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

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
        clang \
        cmake \
        doxygen \
        exuberant-ctags \
        firefox \
        g++-6 \
        gcc-6 \
        gfortran-6 \
        git \
        git-core \
        google-mock \
        graphviz \
        icu-devtools \
        libasound2-dev \
        libblocksruntime-dev \
        libboost-all-dev \
        libbsd-dev \
        libbz2-dev \
        libcurl4-openssl-dev \
        libedit-dev \
        libeigen3-dev \
        libfftw3-dev \
        libflac-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgoogle-glog0v5 \
        libgtest-dev \
        libicu-dev \
        liblzma-dev \
        libncurses5-dev \
        libogg-dev \
        libpython-dev \
        libsndfile-dev \
        libsqlite3-dev \
        libsox-fmt-all \
        libtool \
        libvorbis-dev \
        libxml2-dev \
        lsyncd \
        mesa-common-dev \
        ninja-build \
        parallel \
        pkg-config \
        python \
        python3-pip \
        rsync \
        software-properties-common \
        sudo \
        systemtap-sdt-dev \
        swig \
        tigervnc-standalone-server \
        ubuntu-drivers-common \
        uuid-dev \
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
    chown mluser:mluser $HOME &&\
    mkdir $HOME/downloads && chown mluser:mluser $HOME/downloads && chmod 777 $HOME/downloads

USER mluser
WORKDIR $HOME

# python, pip and conda
RUN pip3 install --upgrade pip
ENV PATH="$HOME/anaconda3/bin:$PATH"
RUN cd $HOME/downloads && \
    wget -q https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh && \
    bash $HOME/downloads/Anaconda3-2019.07-Linux-x86_64.sh -b && \
    conda update -y -n base -c defaults conda && \
    cd $HOME && rm -rf $HOME/downloads

COPY files/ $HOME/files/
RUN sudo chown -R mluser:mluser $HOME &&\
    cp $HOME/files/.bashrc $HOME &&\
    cp $HOME/files/.exrc $HOME &&\
    cp $HOME/files/.vimrc $HOME &&\
    mkdir $HOME/.ssh && cp $HOME/files/ssh_config $HOME/.ssh/config &&\
    mkdir $HOME/.jupyter && cp $HOME/files/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

# conda, python packages, environment.yml, jupyter
RUN conda clean -y -a && \
    conda env create -f $HOME/files/environment.yml && \
    conda init bash
ENV PATH /usr/local/anaconda3/bin:$PATH
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda activate ml && jupyter contrib nbextension install --user && \
    echo "conda activate ml" >> ~/.bashrc && \
    mkdir $HOME/dev && \
    mkdir $HOME/scripts && cp $HOME/files/ju.sh $HOME/scripts && chmod 777 $HOME/scripts/*

# Install docker cli on the docker image
ENV curl_path=$HOME/anaconda3/envs/ml/bin/curl
ENV DOCKERVERSION=18.06.3-ce
ENV docker_url=https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz
RUN sudo ${curl_path} -fsSLO $docker_url && \
    sudo tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 -C /usr/local/bin docker/docker && \
    sudo rm docker-${DOCKERVERSION}.tgz

EXPOSE 8888
SHELL ["/bin/bash", "-c"]

RUN conda install -c fastai -c pytorch -c anaconda fastai gh anaconda

COPY lib/utils/ $HOME/lib/utils
RUN sudo chown -R mluser:mluser $HOME/lib && \
    pip install -e $HOME/lib/utils

COPY lib/nn2/ $HOME/lib/nn2
RUN sudo chown -R mluser:mluser $HOME/lib/nn2 && \
    pip install -e $HOME/lib/nn2