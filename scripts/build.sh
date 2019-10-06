
#
# ml
# Build and upload
#
# RUN: ~/dev/lib/ml/scripts/build.sh

cd ~/dev/lib/ml && \
docker build -t ml . && \
docker tag ml digitallogic/private:ml && \
docker rm -f ml-jupyter 2> /dev/null
docker run -dit \
    --name ml-jupyter \
    --restart unless-stopped \
    --privileged \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml /bin/bash scripts/ju.sh

exit 0

docker push digitallogic/private:ml


#--FORCE_COPY=True
#--no-cache