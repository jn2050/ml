# ml build and launch
#
# RUN: ~/dev/lib/ml/scripts/build.sh
# RUN: ml-build

# Build ml image on mac
cd ~/dev/lib/ml
docker build -t ml . && \
docker tag ml digitallogic/private:ml
docker push digitallogic/private:ml

# Build ml image on cuda1
# cd ~/lib/ml && sudo docker build -t ml .
# sudo docker push digitallogic/private:ml
## sudo docker run -it --rm ml /bin/bash

# Launch ml-jupyter on mac
sleep 20
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

# Launch ml-jneto-jupyter on cuda1
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io /bin/bash -c "
    (sudo docker rm -f ml-jneto-jupyter 2> /dev/null || true) &&\
    sudo docker pull digitallogic/private:ml &&\
    sudo docker run -d \
        --name ml-jneto-jupyter \
        --restart unless-stopped \
        --gpus all \
        --privileged \
        -p 8001:8888 \
        -v /home/jneto/dev:/users/ml/dev \
        -v /dataf:/users/ml/dev/data \
        digitallogic/private:ml /bin/bash scripts/ju.sh
"

exit 0

# Launch ml-pneto-jupyter on cuda1
(sudo docker rm -f ml-pneto-jupyter 2> /dev/null || true) &&\
sudo docker pull digitallogic/private:ml &&\
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


# Environment with db
# docker stack rm ml
# docker stack deploy -c /Users/jneto/dev/lib/ml/docker-compose-db-test.yml ml
# docker exec -it `docker ps | grep ml_db | awk '{print $1}'` /usr/bin/psql -U postgres
# docker stack rm ml

# docker system prune --all
# --no-cache