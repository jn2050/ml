# ml deploy
# RUN: $HOME/dev/lib/ml/scripts/build.sh
# ALIAS: ml-build

# Build ml image on cuda1
cd ~/dev/lib/ml && git add . && git commit -m 'update' && git push &&\
ssh -i ~/.ssh/jn2020 -p 9021 jneto@cuda1.dl2050.com \
    "cd ~/lib && rm -rf ml && git clone https://github.com/jn2050/ml.git && cd ml &&\
    sudo docker build -t ml . &&\
    sudo docker tag ml digitallogic/private:ml &&\
    sudo docker push digitallogic/private:ml"
exit 0

# sudo docker run -it --rm ml /bin/bash
# --no-cache