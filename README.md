# ml: Docker setup for Machine Learning environment
* Includes Python machine learning, pythorch, fast.ai
  * Currently CPU only
  * "evolve" means unleash your greatness to be better and better. "LV" are capitalized to acknowledge the nurturing of Learning and Vision ([LV](http://www.lv-nus.org)) group, Nation University of Singapore (NUS).
* This work was done during Jian Zhao served as a short-term "Texpert" Research Scientist at Tencent FiT DeepSea AI Lab, Shenzhen, China.

|Author|João Neto|
|:---:|:---:|

****
### Pre-Requisites 

* Install Docker on macOS
* Setup Docker to use half of RAM and half of CPUs


****
### Instalation 

1. https://github.com/jn2050/ml.git
2. cd ml
3. docker build -t ml .

****
### Usage 

* Bash shell:

```
docker run -it --rm \
    -v <your_dev_dir>:/users/mluser/dev \
    -v <your_data_dir>:/users/mluser/data \
    ml /bin/bash
```

* Launch Jupyer :

```
docker run -dit \
    --restart unless-stopped \
    --cap-add SYS_PTRACE \
    -p 8888:8888 \
    -v /Users/jneto/dev:/users/mluser/dev \
    -v /Users/jneto/data:/users/mluser/data \
    ml jupyter notebook
```

Survives Mac reboots. To stop container run:

```
docker stop ml
```
