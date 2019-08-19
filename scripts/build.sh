
# Setup
# aws ecr create-repository --repository-name ml

# Purge
# docker rm $(docker ps -a -q) && docker rmi $(docker images -q) && docker rmi $(docker images -q) --force

cd /Users/jneto/dev/proj/ml
docker build -t ml . # --no-cache


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
    ml /bin/bash scripts/ju.sh
#docker rm -f ID


# Launch on remote server

docker login
$(aws ecr get-login --no-include-email --region eu-west-1) &&\
    docker tag ml 925630183784.dkr.ecr.eu-west-1.amazonaws.com/ml && \
    docker push 925630183784.dkr.ecr.eu-west-1.amazonaws.com/ml

ssh -i ~/.ssh/id_rsa_cuda joaoneto@10.0.0.157 \
    "sudo $(aws ecr get-login --no-include-email --region eu-west-1) && \
    sudo docker pull 925630183784.dkr.ecr.eu-west-1.amazonaws.com/ml"












docker build -t ml2 -f swift.Dockerfile .

docker run -it --rm \
    --cap-add SYS_PTRACE \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml2 /bin/bash