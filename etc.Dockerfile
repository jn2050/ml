
# Install docker cli
# ENV curl_path=$HOME/anaconda3/envs/ml/bin/curl
# ENV DOCKER_VERSION=18.06.3-ce    
# ENV docker_url=https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz
# RUN sudo ${curl_path} -fsSLO $docker_url && \
#     sudo tar xzvf docker-${DOCKER_VERSION}.tgz --strip 1 -C /usr/local/bin docker/docker && \
#     sudo rm docker-${DOCKER_VERSION}.tgz


    # volumes:
    #   - /var/run/docker.sock:/var/run/docker.sock


# RUN conda install -y -c fastai -c pytorch -c anaconda fastai gh anaconda
# RUN pip install -U fastai
# https://github.com/fastai/docker-containers/blob/master/fastai-build/Dockerfile
# RUN git clone https://github.com/fastai/fastcore &&\
#     pip install -e "fastcore[dev]" &&\
#     git clone https://github.com/fastai/fastai &&\
#     pip install -e "fastai[dev]"

RUN pip install -U dlogicutils

COPY lib/nn2/ $HOME/lib/nn2
RUN sudo chown -R mluser:mluser $HOME/lib/nn2 && \
    pip install -e $HOME/lib/nn2

# RUN sudo apt update &&\
#     sudo apt -y install nodejs &&\
#     sudo apt -y install npm &&\
#     sudo npm install -g ijavascript &&\
#     ijsinstall

# COPY --chown=ml:ml lib/nn2/ $HOME/lib/nn2
# RUN pip install -e $HOME/lib/nn2
