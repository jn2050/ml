#
# ml deploy
#

# ml-jupyter on mac
docker rm -f ml-jupyter 2> /dev/null
docker run -dit \
    --name ml-jupyter \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /Users/jneto/dev:/users/ml/dev \
    -v /Users/jneto/data:/users/ml/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh
# docker rm -f ml-jupyter

# ml-sh on mac
docker run -it --rm \
    --name ml-sh \
    --privileged \
    -v /Users/jneto/dev:/users/ml/dev \
    -v /Users/jneto/data:/users/ml/data \
    digitallogic/private:ml /bin/bash

# Pull
sudo docker login
sudo docker pull digitallogic/private:ml

# ml-jneto-jupyter on cuda1
sudo docker run -d \
    --name ml-jneto-jupyter \
    --restart unless-stopped \
    --gpus all \
    --privileged \
    -p 8001:8888 \
    -v /home/jneto/dev:/users/ml/dev \
    -v /dataf:/users/ml/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh
# sudo docker rm -f ml-jneto-jupyter

# ml-sh on cuda1
sudo docker run -it --rm \
    --name ml-jneto-sh \
    --privileged \
    -v /home/jneto/dev:/users/ml/dev \
    -v /dataf:/users/ml/data \
    digitallogic/private:ml /bin/bash

# ml-pneto-jupyter on cuda1
sudo docker run -d \
    --name ml-pneto-jupyter \
    --restart unless-stopped \
    --gpus all \
    --privileged \
    -p 9000:8888 \
    -v /home/pneto:/users/ml/dev \
    -v /dataf:/users/ml/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh
# sudo docker rm -f ml-pneto-jupyter


# alias cuda='ssh -i ~/.ssh/pneto_cuda -p 9022 -L 9000:localhost:9000 pneto@ml.dlogic.io'
# rsync -zrave 'ssh -i ~/.ssh/pneto_cuda -p 9022' pneto@ml.dlogic.io:/home/pneto/file .
