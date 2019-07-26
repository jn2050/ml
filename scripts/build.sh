docker build -t ml .
docker build --no-cache -t ml .

docker run -it --rm \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash

docker run -dit \
    --restart unless-stopped \
    --cap-add SYS_PTRACE \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml jupyter notebook

docker rm $(docker ps -a -q) && docker rmi $(docker images -q) && docker rmi $(docker images -q) --force

docker build -t ml2 -f swift.Dockerfile .

docker run -it --rm \
    --cap-add SYS_PTRACE \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml2 /bin/bash