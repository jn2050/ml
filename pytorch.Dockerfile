FROM digitallogic/private:ml
RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia