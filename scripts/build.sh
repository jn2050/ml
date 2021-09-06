# ml deploy
# RUN: ~/dev/lib/ml/scripts/deploy.sh
# RUN: ml-build

# Build ml image on cuda1
cd ~/dev/lib/ml
git add . && git commit -m 'update' && git push
ssh -i ~/.ssh/jn2020 -p 9022 jneto@ml.dlogic.io \
    "cd ~/lib && rm -rf ml && git clone https://github.com/jn2050/ml.git && cd ml &&\
    sudo docker build -t ml . &&\
    sudo docker tag ml digitallogic/private:ml &&\
    sudo docker push digitallogic/private:ml"