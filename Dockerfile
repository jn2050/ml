FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM ubuntu:18.04

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
    mkdir $HOME/downloads && chown mluser:mluser $HOME/downloads 

USER mluser
WORKDIR $HOME

COPY files/ $HOME/files/
RUN sudo chown mluser:mluser $HOME && sudo chown -R mluser:mluser $HOME
RUN cp $HOME/files/.bashrc $HOME &&\
    cp $HOME/files/.exrc $HOME &&\
    cp $HOME/files/.vimrc $HOME &&\
    cp $HOME/files/.inputrc $HOME &&\
    mkdir $HOME/.ssh && cp $HOME/files/ssh_config $HOME/.ssh/config &&\
    mkdir $HOME/.jupyter && cp $HOME/files/jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py
RUN git clone https://github.com/VundleVim/Vundle.vim.git $HOME/.vim/bundle/Vundle.vim && \
    cd $HOME/downloads && git clone https://github.com/universal-ctags/ctags.git
#     cd ctags && ./autogen.sh && ./configure && \
#     make && sudo make install

RUN pip3 install --upgrade pip

ENV PATH="$HOME/anaconda3/bin:$PATH"
RUN cd $HOME/downloads && wget -q https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh && \
    bash $HOME/downloads/Anaconda3-2019.07-Linux-x86_64.sh -b && \
    conda update -y -n base conda

RUN conda env create -f $HOME/files/environment.yml
RUN conda install -y -c pytorch -c fastai fastai

RUN conda init bash
ENV PATH /usr/local/anaconda3/bin:$PATH

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda activate ml && jupyter contrib nbextension install --user && \
    echo "conda activate ml" >> ~/.bashrc && \
    mkdir $HOME/dev && \
    mkdir $HOME/scripts && echo "cd $HOME/dev && jupyter notebook" > $HOME/scripts/ju.sh
EXPOSE 8888

# RUN pip install awscli --upgrade --user

ENV swift_tf_version=swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
#ENV swift_tf_url=https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/
ENV swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/$swift_tf_version
RUN cd $HOME/downloads && \
    wget -q $swift_tf_url && \
    tar xf $swift_tf_version
RUN mkdir $HOME/swift && mv $HOME/downloads/usr $HOME/swift && \
    echo 'export PATH=$HOME/swift/usr/bin:$PATH' >> $HOME/.bashrc

ENV PATH="$HOME/swift/usr/bin:$PATH"
RUN mkdir $HOME/git && cd $HOME/git && \
    git clone https://github.com/google/swift-jupyter.git && \
    cd $HOME/git/swift-jupyter && \
    python register.py --sys-prefix --swift-python-use-conda --use-conda-shared-libs   --swift-toolchain $HOME/swift


SHELL ["/bin/bash", "-c"]
COPY lib/utils/ $HOME/lib/utils
RUN sudo chown -R mluser:mluser $HOME/lib && \
    pip install -e $HOME/lib/utils
