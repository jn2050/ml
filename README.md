# ml: docker setup for machine learning environment
* Includes Python machine learning, pythorch, fast.ai and swift for tensorflow


|Author|João Neto|
|:---:|:---:|

****
### Pre-Requisites 

* Install Docker on macOS
* Setup Docker to use half of RAM and half of CPUs


****
### Instalation from source

```
git clone https://github.com/jn2050/ml.git
cd ml
sudo docker build -t ml .
```


****
### Download from dockerhub

```
docker login
docker pull digitallogic/private:ml
```


****
### Usage 

* Bash shell:

```
docker run -it --rm \
    --privileged \
    -v <dev_dir>:/users/mluser/dev \
    -v <data_dir>:/users/mluser/data \
    ml /bin/bash
```

* Launch Jupyer (survives reboots):

```
docker run -dit \
    --privileged \
    --restart unless-stopped \
    -p 8888:8888 \
    -v <dev_dir>:/users/mluser/dev \
    -v <data_dir>:/users/mluser/data \
    ml jupyter notebook
```


To stop container:

```
docker stop ml
```

To stop and remove image:

```
docker ps
docker rm -f <container_id>
```