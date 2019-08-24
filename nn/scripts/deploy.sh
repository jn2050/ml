#!/usr/bin/env bash

zip -r nn.zip app lib prep scripts *.py *.txt
rsync -rave 'ssh -p 9022 -i ~/.ssh/id_rsa_cuda' \
    nn.zip \
    jneto@ml.dlogic.io:/home/jneto/proj
ssh -i ~/.ssh/id_rsa_cuda jneto@ml.dlogic.io -p 9022 \
    "cd proj; rm -rf nn/*; mv nn.zip nn; cd nn; unzip nn.zip; rm nn.zip"

#EEG
#cp /Users/jneto/ml/Projects/EEG/EEG.xlsx /Users/jneto/data/eeg
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' mat.zip jneto@ml.dlogic.io:/home/jneto/dataf/eeg
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' *.csv jneto@ml.dlogic.io:/home/jneto/dataf/eeg
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' /Users/jneto/data/eeg/EEG.xlsx jneto@ml.dlogic.io:/home/jneto/dataf/eeg
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' jneto@ml.dlogic.io:/home/jneto/dataf/eeg/models/* /Users/jneto/data/eeg/models/
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' jneto@ml.dlogic.io:/home/jneto/dataf/eeg/models/model_EEG_Net_best_best.pth.tar /Users/jneto/data/eeg/models/model_EEG_Net_best.pth.tar

#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' f.zip jneto@ml.dlogic.io:/home/jneto/dataf/caravela
#rsync -rave 'ssh -p 9022 -i ~/.ssh/jn.pem' jneto@ml.dlogic.io:/home/jneto/dataf/caravela/models/apolices_results.feather /Users/jneto/ML/projects/Caravela