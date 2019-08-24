#!/usr/bin/env bash
ssh -i ~/.ssh/jneto jneto@ml.dlogic.io -p 9022 "cd dataf/style/output; zip output.zip *.jpg"
cd /Users/jneto/data/style/output
rsync -rave 'ssh -p 9022 -i ~/.ssh/jneto' jneto@ml.dlogic.io:/home/jneto/dataf/style/output/output.zip .
unzip -o output.zip
rm output.zip