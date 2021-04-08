# ml deploy
# RUN: ~/dev/lib/ml/scripts/deploy.sh
# RUN: ml-deploy

# Build ml image on cuda1
cd ~/dev/lib/ml
git add . && git commit -m 'update' || git push
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io \
    "cd ~/lib && rm -rf ml && git clone https://github.com/jn2050/ml.git && cd ml &&\
    sudo docker build -t --no-cache ml . &&\
    sudo docker tag ml digitallogic/private:ml &&\
    sudo docker push digitallogic/private:ml" &&\
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io \
    "sudo docker rm -f ml-jneto-jupyter 2> /dev/null"
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io \
    "sudo docker pull digitallogic/private:ml &&\
    sudo docker run -d \
        --name ml-jneto-jupyter \
        --restart unless-stopped \
        --gpus all \
        --privileged \
        -p 8001:8888 \
        -v /home/jneto/dev:/users/ml/dev \
        -v /dataf:/users/ml/dev/data \
        digitallogic/private:ml /bin/bash scripts/ju.sh" &&\
docker pull digitallogic/private:ml &&\
docker rm -f ml-jupyter 2> /dev/null &&\
docker run -dit \
    --name ml-jupyter \
    --restart unless-stopped \
    -p 8888:8888 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /Users/jneto/dev:/users/ml/dev \
    -v /Users/jneto/data:/users/ml/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh

# --network test
# ml-sh on mac
# docker run -it --rm --name ml-sh -v ~/dev:/users/ml/dev -v ~/data:/users/ml/data digitallogic/private:ml /bin/bash

exit 0

# Environment with db
# docker network create --driver bridge test
docker run -d \
    --name dbtest \
    --network test \
    -p 5432:5432 \
    -e POSTGRES_DB=postgres \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=rootroot postgres \
    -v olap:/data
    postgres

exit 0

# Launch ml-jneto-jupyter on cuda1
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io /bin/bash -c "
    (sudo docker rm -f ml-jneto-jupyter 2> /dev/null || true) &&\
    sudo docker pull digitallogic/private:ml &&\
    sudo docker run -d \
        --name ml-jneto-jupyter \
        --restart unless-stopped \
        --gpus all \
        --network test \
        -p 8001:8888 \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /home/jneto/dev:/users/ml/dev \
        -v /dataf:/users/ml/dev/data \
        digitallogic/private:ml /bin/bash scripts/ju.sh
"

# ml-sh on cuda1
# sudo docker run -it --rm --name ml-jneto-sh -v ~/dev:/users/ml/dev -v /dataf:/users/ml/data digitallogic/private:ml /bin/bash

# ml-pneto-jupyter on cuda1
(sudo docker rm -f ml-pneto-jupyter 2> /dev/null || true) &&\
sudo docker pull digitallogic/private:ml &&\
sudo docker run -d \
    --name ml-pneto-jupyter \
    --restart unless-stopped \
    --gpus all \
    -p 9000:8888 \
    -v /home/pneto:/users/ml/dev \
    -v /dataf:/users/ml/dev/data \
    digitallogic/private:ml /bin/bash scripts/ju.sh
# sudo docker rm -f ml-pneto-jupyter

# alias cuda='ssh -i ~/.ssh/pneto_cuda -p 9022 -L 9000:localhost:9000 pneto@ml.dlogic.io'
# rsync -zrave 'ssh -i ~/.ssh/pneto_cuda -p 9022' pneto@ml.dlogic.io:/home/pneto/file .

# Build ml image on mac
# cd ~/dev/lib/ml
# docker build -t ml . && \
# docker tag ml digitallogic/private:ml
# docker push digitallogic/private:ml

# Build ml image on cuda1
# cd ~/lib/ml && sudo docker build -t ml .
# sudo docker tag ml digitallogic/private:ml
# sudo docker push digitallogic/private:ml
## sudo docker run -it --rm ml /bin/bash

# sudo docker login
# docker system prune --all
# --no-cache