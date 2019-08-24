"""
    DNN Classifier
    jneto, 2017.12

    Requeries Visdom for image display:
    conda install -c conda-forge visdom
    python -m visdom.server

    Run with:
    CUDA_VISIBLE_DEVICES=0,1 python main.py --path dogs

"""

import torch
from lib.config import Config
from app.rouen import rouen
from app.eeg import eeg
from app.auto import auto
from app.dogs import dogs
from app.retina import retina
from app.tab import tab
from app.colab import colab
from app.bitcoin import bitcoin
from app.style import style
from app.gan import gan
from app.caravela import caravela


META = {
    'rouen': rouen,
    'eeg': eeg,
    'auto': auto,
    'dogs': dogs,
    'retina': retina,
    'tab': tab,
    'colab': colab,
    'bitcoin': bitcoin,
    'style': style,
    'gan': gan,
    'caravela': caravela
}


def main():
    cfg = Config().parse()
    META[cfg.dir](cfg)
    exit(0)


if __name__ == '__main__':
    main()
