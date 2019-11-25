
#
# ml
# Build and upload
#
# RUN: ~/dev/lib/ml/scripts/build.sh

cd ~/dev/lib/ml
docker stack rm ml
docker build -t ml . && \
docker tag ml digitallogic/private:ml
sleep 20
docker stack deploy -c /Users/jneto/dev/lib/ml/docker-compose-db-test.yml ml

exit 0

docker push digitallogic/private:ml

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