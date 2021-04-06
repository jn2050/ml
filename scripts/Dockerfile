FROM debian:10.9

RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get update &&\
    apt-get install -y --fix-missing \
        sudo wget curl rsync vim-nox python3-pip git git-core iputils-ping

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

RUN jupyter contrib nbextension install --user

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skip_cache
RUN pip install dl2050utils
