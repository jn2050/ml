
# db server
(sudo docker rm -f db || true) &&\
sudo docker run -d --name db\
    --restart unless-stopped \
    --network webnet \
    -e POSTGRES_DB=postgres \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=rootroot \
    -e PGDATA=/data/db \
    -v db:/data \
    postgres:13.3