
docker volume create ml-db-pg

docker-machine create \
    --driver generic \
    --generic-ip-address=62.28.30.230 \
    --generic-ssh-port=9022 \
    --generic-ssh-user jneto \
    --generic-ssh-key ~/.ssh/id_rsa_cuda \
    cuda1

docker-machine regenerate-certs --client-certs cuda1
docker-machine ls

docker-machine stop cuda1
docker-machine start cuda1
docker-machine rm cuda1

eval $(docker-machine env cuda1)
eval $(docker-machine env -u)

docker-machine -D ssh cuda1