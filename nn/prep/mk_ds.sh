#!/usr/bin/env bash

cp ~/data/retina/train.csv ~/data/retina/train_all.csv
cp ~/data/retina/valid.csv ~/data/retina/valid_all.csv

python nn.py retina --mksamples
cp ~/data/retina/samples.csv ~/data/retina/train.csv

python nn.py retina --mksamples
cp ~/data/retina/samples.csv ~/data/retina/valid.csv