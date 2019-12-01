
# cuda1 sh

# ssh -i ~/.ssh/id_rsa_cuda -p 9022 -L 8001:localhost:8001 jneto@ml.dlogic.io

# -p or --publish: host:container

sudo docker pull digitallogic/private:ml

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
    -p 8001:8888 \
    -v /home/jneto/dev:/users/mluser/dev \
    -v /dataf:/users/mluser/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh


# cuda1 stop jupyter
sudo docker rm -f ml-jneto-jupyter

# rsync
rsync -rave 'ssh -i ~/.ssh/id_rsa_cuda -p 9022' \
    /Users/jneto/data/caravela/stage/df_nn* \
    jneto@ml.dlogic.io:/dataf/caravela