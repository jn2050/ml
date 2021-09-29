# ml deploy
# RUN: ~/dev/lib/ml/scripts/deploy.sh
# ALIAS: ml-build

# Build ml image on cuda1
cd ~/dev/lib/ml
git add . && git commit -m 'update' && git push
ssh -i ~/.ssh/jn2020 -p 9021 jneto@cuda1.dl2050.com \
    "cd ~/lib && rm -rf ml && git clone https://github.com/jn2050/ml.git && cd ml &&\
    sudo docker build --no-cache -t ml . &&\
    sudo docker tag ml digitallogic/private:ml &&\
    sudo docker push digitallogic/private:ml"
exit 0

# --no-cache
# export PATH="~/scripts:$PATH"


# Build pythorch image on cuda
cd ~/lib/ml &&\
sudo docker build -f pytorch.Dockerfile -t ml-pytorch . &&\
sudo docker tag ml-pytorch digitallogic/private:ml-pytorch &&\
sudo docker push digitallogic/private:ml-pytorch