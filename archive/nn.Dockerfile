#  Alpine based nn Docker file
#  Based on Miniconda Python 3.5 Docker image
#      https://github.com/frol/docker-alpine-miniconda3
#

FROM frolvlad/alpine-glibc:alpine-3.8

RUN apk add bash bash-doc bash-completion
RUN apk add util-linux pciutils usbutils coreutils binutils findutils grep
RUN apk add build-base

ENV CONDA_DIR="/opt/conda"
ENV PATH="$CONDA_DIR/bin:$PATH"

# Install conda
RUN CONDA_VERSION="4.5.4" && \
    CONDA_MD5_CHECKSUM="a946ea1d0c4a642ddf0c3a26a18bb16d" && \
    apk add --no-cache --virtual=.build-dependencies wget ca-certificates bash && \
    mkdir -p "$CONDA_DIR" && \
    wget "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" -O miniconda.sh && \
    echo "$CONDA_MD5_CHECKSUM  miniconda.sh" | md5sum -c && \
    bash miniconda.sh -f -b -p "$CONDA_DIR" && \
    echo "export PATH=$CONDA_DIR/bin:\$PATH" > /etc/profile.d/conda.sh && \
    rm miniconda.sh && \
    conda update --all --yes && \
    conda config --set auto_update_conda False && \
    rm -r "$CONDA_DIR/pkgs/" && \
    apk del --purge .build-dependencies && \
    mkdir -p "$CONDA_DIR/locks" && \
    chmod 777 "$CONDA_DIR/locks"

# Install OpenCV3 Python bindings
# From: https://github.com/anibali/docker-pytorch/blob/master/no-cuda/Dockerfile
#RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#    libgtk2.0-0 \
#    libcanberra-gtk-module \
# && sudo rm -rf /var/lib/apt/lists/*
#RUN conda install -y -c menpo opencv3=3.1.0 \
# && conda clean -ya
#RUN apk update

#RUN conda update -n base -c defaults conda
RUN conda install psutil
RUN conda install -c anaconda numpy pandas scipy seaborn pillow flask
RUN conda install scikit-learn
RUN conda install -c conda-forge librosa
RUN conda install pytorch torchvision -c pytorch

RUN conda install jupyter -y
RUN conda install -c conda-forge jupyter_contrib_nbextensions -y
RUN jupyter contrib nbextension install --user
RUN conda install openpyxl

RUN pip install SQLAlchemy
RUN pip install pymysql PyMySQL
 
RUN pip install --ignore-installed awscli boto3 rfc3339
RUN pip install bcrypt PyJWT
RUN pip install starlette uvicorn starlette-jwt aiofiles
RUN pip install -U feather-format

COPY . /nn
EXPOSE 8008

# CMD jupyter notebook \
#     --ip=0.0.0.0 \
#     --port=8008 \
#     --NotebookApp.token='' \
#     --NotebookApp.password='' \
#     --NotebookApp.iopub_data_rate_limit=10000000000 \
#     --allow-root \
#     --notebook-dir=/nn \
#     --no-browser