################################################################################
# NN AWS Setup
################################################################################


########################################
# ERS - Docker repository
########################################

aws ecr create-repository --repository-name nn

# Cleanup
aws ecr delete-repository --force --repository-name nn


########################################
# EFS
########################################

aws ec2 create-security-group \
--region eu-west-1 \
--group-name efs-ec2-sg \
--description "EFS SG for EC2 client instances" \
--vpc-id vpc-0f00cd0f141567ec1
# -> sg-0edfb4cd538cc1109

aws ec2 authorize-security-group-ingress \
--region eu-west-1
--group-id sg-0edfb4cd538cc1109 \
--protocol tcp \
--port 22 \
--cidr 0.0.0.0/0 \

aws ec2 create-security-group \
--region eu-west-1 \
--group-name efs-mt-sg \
--description "Amazon EFS SG for mount target" \
--vpc-id vpc-0f00cd0f141567ec1
# -> sg-0eddfad55270af9cd

aws ec2 authorize-security-group-ingress \
--region eu-west-1 \
--group-id sg-0eddfad55270af9cd \
--protocol tcp \
--port 2049 \
--source-group sg-0edfb4cd538cc1109

# Open all for debug
aws ec2 authorize-security-group-ingress \
--region eu-west-1 \
--group-id sg-0eddfad55270af9cd \
--protocol tcp \
--port 2049 \
--cidr 0.0.0.0/0

# Revoke all
aws ec2 revoke-security-group-ingress \
--region eu-west-1 \
--group-id sg-0eddfad55270af9cd

aws efs create-file-system \
--region eu-west-1 \
--creation-token nn-efs
# -> fs-749928bc

aws efs create-tags \
--region eu-west-1 \
--file-system-id fs-749928bc \
--tags Key=Name,Value=nn-efs

aws efs create-mount-target \
--region eu-west-1 \
--file-system-id fs-749928bc \
--subnet-id subnet-061ed066bb7b93130 \
--security-group sg-0eddfad55270af9cd
# -> fsmt-41220c88

# Cleanup

aws efs delete-mount-target \
--region eu-west-1 \
--mount-target-id fsmt-41220c88

aws efs delete-file-system \
--region eu-west-1 \
--file-system-id fs-749928bc


########################################
# Mount EFS inside E2C -> Dockerfile
########################################

docker-machine ssh persan1
sudo apt-get update
sudo apt-get -y install nfs-common
sudo apt install -y awscli
mkdir /home/ubuntu/data
sudo mount -t nfs \
    -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport \
    fs-749928bc.efs.eu-west-1.amazonaws.com:/ \
    /home/ubuntu/data
sudo chmod go+rw /home/ubuntu/data

AWS_ACCESS_KEY_ID=AKIAILMXTEJI7VMA2G5Q \
    AWS_SECRET_ACCESS_KEY=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 \
    aws s3 cp s3://dlogic-nn/data/eeg /home/ubuntu/data/eeg --recursive

########################################
# S3
########################################

aws s3 create-bucket \
    --region eu-west-1 \
    --create-bucket-configuration LocationConstraint=eu-west-1 \
    --bucket dlogic-nn

aws s3 cp ~/data/eeg s3://dlogic-nn/data/eeg --recursive

# Cleanup
aws s3 rm s3://dlogic-nn --recursive
aws s3 rb s3://dlogic-nn --force
