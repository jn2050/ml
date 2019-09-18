
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

docker run -dit \
    --restart unless-stopped \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh
#docker rm -f ID

docker run -it --rm \
    --privileged \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash


docker tag ml digitallogic/private:ml && \
    docker push digitallogic/private:ml


docker pull digitallogic/private:ml
