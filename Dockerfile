#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:11.1-devel-ubuntu20.04
FROM nvidia/cuda:11.4.1-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get update &&\
    apt-get install -y --fix-missing \
        sudo bash wget curl rsync vim-nox uuid-dev python python3-pip git git-core \
        locate ffmpeg libsm6 libxext6 iputils-ping postgresql-client
ENV TZ=Europe/Lisbon
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]

ENV PATH="$HOME/anaconda3/bin:$PATH"
# ENV ANACONDA_VER=Anaconda3-2020.11-Linux-x86_64.sh
ENV ANACONDA_VER=Anaconda3-2021.05-Linux-x86_64.sh
RUN pip3 install --upgrade pip
RUN cd $HOME/downloads &&\
    wget -q https://repo.anaconda.com/archive/$ANACONDA_VER &&\
    bash $ANACONDA_VER -b &&\
    rm $ANACONDA_VER &&\
    conda update -y -n base -c defaults conda &&\
    conda init bash
RUN conda env create -f $HOME/files/environment.yml &&\
    echo "conda activate ml" >> ~/.bashrc
RUN jupyter contrib nbextension install --user

# RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

RUN pip install --upgrade google-cloud-storage

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skip_cache
RUN pip install --upgrade dl2050utils
RUN pip install --upgrade dl2050nn
EXPOSE 8888
