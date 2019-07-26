FROM ml

# swift
# wget https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
# wget https://storage.googleapis.com/swift-tensorflow-artifacts/releases/v0.4/rc2/swift-tensorflow-RELEASE-0.4-ubuntu18.04.tar.gz

RUN cd $HOME/downloads &&\
    sudo wget https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz &&\
    sudo apt-fast -y install git cmake ninja-build clang python uuid-dev libicu-dev icu-devtools libbsd-dev \
        libedit-dev libxml2-dev libsqlite3-dev swig libpython-dev libncurses5-dev pkg-config \
        libblocksruntime-dev libcurl4-openssl-dev systemtap-sdt-dev tzdata rsync &&\
    sudo tar xf swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
RUN mkdir $HOME/swift && cd $HOME/swift && cp -r $HOME/downloads/usr ./

RUN cd $HOME/git && git clone https://github.com/google/swift-jupyter.git &&\
    cd swift-jupyter &&\
    python register.py --sys-prefix --swift-python-use-conda --use-conda-shared-libs --swift-toolchain $HOME/swift &&\
    cd ~/git && git clone https://github.com/fastai/fastai_docs.git
