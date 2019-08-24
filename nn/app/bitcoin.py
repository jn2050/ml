import pandas as pd
from lib.data import *
from lib.models import *
from lib.learner import Learner
from lib.metrics import *


def bitcoin(cfg):
    df = pd.read_csv(os.path.join(cfg.path, 'bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv'))
    #rint(df.isnull().values.any())


