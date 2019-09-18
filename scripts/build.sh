
# ml
#
#   AWS setup
#   aws ecr create-repository --repository-name ml
#
# PURGE: docker rm $(docker ps -a -q) && docker rmi $(docker images -q) && docker rmi $(docker images -q) --force
# RUN: cd ~/dev/proj/ml && ./scripts/build.sh
#

docker build -t ml .
# --no-cache

# mac jupyter
docker run -dit \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

# mac sh
sudo docker run -it --rm \
    --privileged \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash


# cuda1 jupyter
sudo docker run -d \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

# cuda1 sh
sudo docker run -it --rm \
    --privileged \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/data \
    ml /bin/bash



docker tag ml digitallogic/private:ml && \
    docker push digitallogic/private:ml


docker pull digitallogic/private:ml


# sudo systemctl start docker
# sudo systemctl stop docker

# sudo docker exec -u 0 -it d3f1cc167bc5 /bin/bash
# kill 1