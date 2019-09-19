
#
# ml
#

cd ~/dev/proj/ml
docker build -t ml . # --no-cache

docker tag ml digitallogic/private:ml && \
    docker push digitallogic/private:ml

docker pull digitallogic/private:ml

# mac jupyter
docker run -dit \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

# mac sh
docker run -it --rm \
    --privileged \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash


# cuda1 jupyter (jneto)
sudo docker run -d \
    --restart unless-stopped \
    --privileged \
    --gpus all \
    -p 8000:8888 \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh

# cuda1 sh
sudo docker run -it --rm \
    --privileged \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/data \
    digitallogic/private:ml /bin/bash
