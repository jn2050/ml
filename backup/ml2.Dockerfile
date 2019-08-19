FROM ml

RUN sudo apt-get -y install \
        git \
        cmake \
        ninja-build \
        clang \
        python \
        uuid-dev \
        libicu-dev \
        icu-devtools \
        libbsd-dev \
        libedit-dev \
        libxml2-dev \
        libsqlite3-dev \
        swig \
        libpython-dev \
        libncurses5-dev \
        pkg-config \
        libblocksruntime-dev \
        libcurl4-openssl-dev \
        systemtap-sdt-dev \
        tzdata \
        rsync

ENV swift_tf_version=swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
#ENV swift_tf_url=https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/
ENV swift_tf_url=https://storage.googleapis.com/swift-tensorflow-artifacts/nightlies/latest/$swift_tf_version
RUN cd $HOME/downloads && \
    wget -q $swift_tf_url && \
    tar xf $swift_tf_version
RUN mkdir $HOME/swift && mv $HOME/downloads/swift-tensorflow-toolchain/usr $HOME/swift && \
    echo 'export PATH=$HOME/swift/usr/bin:$PATH' >> $HOME/.bashrc

ENV PATH="$HOME/swift/usr/bin:$PATH"
RUN mkdir $HOME/git && cd $HOME/git && \
    git clone https://github.com/google/swift-jupyter.git && \
    cd $HOME/git/swift-jupyter && \
    python register.py --sys-prefix --swift-python-use-conda --use-conda-shared-libs   --swift-toolchain $HOME/swift
