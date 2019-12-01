
#
# ml
# Build and upload
#
# RUN: ~/dev/lib/ml/scripts/build.sh

cd ~/dev/lib/ml
scripts/cp-nn2.sh
docker stack rm ml
docker build -t ml . && \
docker tag ml digitallogic/private:ml
sleep 20
docker stack deploy -c /Users/jneto/dev/lib/ml/docker-compose-db-test.yml ml
# exit 0

docker push digitallogic/private:ml
ssh -i ~/.ssh/id_rsa_cuda -p 9022 jneto@ml.dlogic.io /home/jneto/scripts/ml.sh
exit 0

docker rm -f ml-jupyter 2> /dev/null
docker run -dit \
    --name ml-jupyter \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

# Environment with db
# docker stack deploy -c /Users/jneto/dev/lib/ml/docker-compose-db-test.yml ml
# docker stack rm ml


# Debug
docker run -it \
    --name ml-jupyter \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash

exit 0