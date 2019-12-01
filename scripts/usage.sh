#
# ml
# Depoy examples
#


# mac sh
docker run -it --rm \
    --name ml-sh \
    --privileged \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash

# mac jupyter
docker run -dit \
    --name ml-jupyter \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

# mac stop jupyter
docker rm -f ml-jupyter




# Pull (optional)
docker pull digitallogic/private:ml

# cuda1 sh
sudo docker run -it --rm \
    --name ml-jneto-sh \
    --privileged \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/data \
    digitallogic/private:ml /bin/bash

# cuda1 jupyter
sudo docker run -d \
    --name ml-jneto-jupyter \
    --restart unless-stopped \
    --privileged \
    --gpus all \
    -p 800o:8888 \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh


# cuda1 stop jupyter
sudo docker rm -f ml-jneto-jupyter
