# ml: docker setup for machine learning environment
* Includes Python machine learning, pythorch, fast.ai and swift for tensorflow
  * Currently CPU only


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
docker pull digitallogic/private:ml
```


****
### Usage 

* Bash shell:

```
docker run -it --rm \
    --privileged \
    -v <your_dev_dir>:/users/mluser/dev \
    -v <your_data_dir>:/users/mluser/data \
    ml /bin/bash
```

* Launch Jupyer :

```
docker run -dit \
    --privileged \
    --restart unless-stopped \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml jupyter notebook
```

Survives Mac reboots.

To stop container run:

```
docker stop ml
```

or to stop and remove image data:

```
docker rm -f <container_id>
```