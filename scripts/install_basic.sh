sudo add-apt-repository -y ppa:apt-fast/stable
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get -y install apt-fast
# prompts

sudo apt-fast -y upgrade

sudo apt-fast install -y ubuntu-drivers-common libvorbis-dev libflac-dev libsndfile-dev cmake build-essential libgflags-dev libgoogle-glog-dev libgtest-dev google-mock zlib1g-dev libeigen3-dev libboost-all-dev libasound2-dev libogg-dev libtool libfftw3-dev libbz2-dev liblzma-dev libgoogle-glog0v5 gcc-6 gfortran-6 g++-6 doxygen graphviz libsox-fmt-all parallel exuberant-ctags vim-nox python-powerline python3-pip
sudo apt-fast install -y tigervnc-standalone-server firefox lsyncd mesa-common-dev ack

cat << 'EOF' >> ~/.ssh/config
Host *
ServerAliveInterval 60
StrictHostKeyChecking no

Host github.com
User git
Port 22
Hostname github.com
TCPKeepAlive yes
IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

echo "alias config='/usr/bin/git --git-dir=$HOME/.cfg/ --work-tree=$HOME'" >> ~/.bashrc
source ~/.bashrc
echo ".cfg" >> .gitignore
git clone --bare https://github.com/fastai/dotfiles.git .cfg/
config checkout
config config --local status.showUntrackedFiles no
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
pip3 install powerline-status

ubuntu-drivers devices
sudo apt-fast install -y nvidia-driver-415 
sudo modprobe nvidia
nvidia-smi

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6   40 --slave /usr/bin/g++ g++ /usr/bin/g++-6 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7   40 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-7

cd
mkdir download
cd download/

wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh -b
cd
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate" >> ~/.bashrc
source ~/.bashrc

cd download
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod u+x cuda_10.0.130_410.48_linux
./cuda_10.0.130_410.48_linux --extract=`pwd`
sudo ./cuda-linux.10.0.130-24817639.run -noprompt
echo /usr/local/cuda-10.0/lib64 | sudo tee -a /etc/ld.so.conf 
sudo ldconfig
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cd ~/download
wget http://files.fast.ai/files/cudnn-10.0-linux-x64-v7.5.0.56.tgz
tar xf cudnn-10.0-linux-x64-v7.5.0.56.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig

cd
mkdir git
cd ~/git
git clone https://github.com/fastai/fastai.git &
git clone https://github.com/fastai/fastprogress.git &
git clone https://github.com/fastai/fastec2.git &
git clone https://github.com/fastai/course-v3.git

sudo snap install hub --classic

cat << 'EOF' >> ~/.bashrc
alias git=hub
alias gpr='git pull-request -m "$(git log -1 --pretty=%B)"'
clonefork() {
hub clone "$1"
cd "${1##*/}"
hub fork
}
EOF
source ~/.bashrc

cd ~/git
conda uninstall -y fastai fastprogress
cd fastai
pip install -e .
cd ../fastprogress
pip install -e .
cd ../fastec2
pip install -e .

jupyter notebook --generate-config
cat << 'EOF' >> ~/.jupyter/jupyter_notebook_config.py
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
EOF
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# swift stuff below

cd ~/download/
wget https://storage.googleapis.com/s4tf-kokoro-artifact-testing/latest/swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
sudo apt-fast -y install git cmake ninja-build clang python uuid-dev libicu-dev icu-devtools libbsd-dev libedit-dev libxml2-dev libsqlite3-dev swig libpython-dev libncurses5-dev pkg-config libblocksruntime-dev libcurl4-openssl-dev systemtap-sdt-dev tzdata rsync
tar xf swift-tensorflow-DEVELOPMENT-cuda10.0-cudnn7-ubuntu18.04.tar.gz
cd
mkdir swift
cd swift
mv ~/download/usr ./
cd
echo 'export PATH=~/swift/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cd ~/git
git clone https://github.com/google/swift-jupyter.git
cd swift-jupyter
python register.py --sys-prefix --swift-python-use-conda --use-conda-shared-libs   --swift-toolchain ~/swift
cd ~/git
git clone https://github.com/fastai/fastai_docs.git
cd fastai_docs/
jupyter notebook
