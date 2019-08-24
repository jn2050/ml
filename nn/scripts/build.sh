
docker build -t nn .
docker tag nn 925630183784.dkr.ecr.eu-west-1.amazonaws.com/nn
$(aws ecr get-login --no-include-email --region eu-west-1)
docker push 925630183784.dkr.ecr.eu-west-1.amazonaws.com/nn

# docker run -it --rm -p 8008:8008 nn /bin/bash
# docker run -d -p 8008:8008 nn
