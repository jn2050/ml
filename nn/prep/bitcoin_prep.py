import pandas as pd
import os
from datetime import date


def bitcoin_prep(path, fname):
    df = pd.read_csv(os.path.join(path, fname))
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    daily = group['Weighted_Price'].mean()
    print(daily.tail())

    d0 = date(2016, 1, 1)
    d1 = date(2017, 10, 15)
    delta = d1 - d0
    days_look = delta.days + 1
    print(days_look)

    d0 = date(2017, 8, 21)
    d1 = date(2017, 10, 20)
    delta = d1 - d0
    days_from_train = delta.days + 1
    print(days_from_train)

    d0 = date(2017, 10, 15)
    d1 = date(2017, 10, 20)
    delta = d1 - d0
    days_from_end = delta.days + 1
    print(days_from_end)


path = os.path.join(os.environ.get('DATA_PATH'), 'bitcoin')
bitcoin_prep(path, 'bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv')