import pandas as pd
import numpy as np
from pandas_summary import DataFrameSummary
import os
import re
from isoweek import Week
import sys
sys.path.append(".")
from structured import *


def prep_elapsed(df):
    columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
    df = df[columns]
    for c in ['SchoolHoliday', 'StateHoliday', 'Promo']:
        df = df.sort_values(['Store', 'Date'])
        get_elapsed(df, c, 'After')
        df = df.sort_values(['Store', 'Date'], ascending=[True, False])
        get_elapsed(df, c, 'Before')
    df = df.set_index("Date")
    columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
    for o in ['Before', 'After']:
        for p in columns:
            a = o + p
            df[a] = df[a].fillna(0)
    bwd = df[['Store'] + columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
    fwd = df[['Store'] + columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum()
    bwd.drop('Store', 1, inplace=True)
    bwd.reset_index(inplace=True)
    fwd.drop('Store', 1, inplace=True)
    fwd.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
    df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
    df.drop(columns, 1, inplace=True)
    df["Date"] = pd.to_datetime(df.Date)
    return df


path = os.path.join(os.environ.get('DATA_PATH'), 'ross') + '/'

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(f'{path}{fname}.csv', low_memory=False) for fname in table_names]
# for t in tables: print(DataFrameSummary(t).summary())
train, store, store_states, state_names, googletrend, weather, test = tables

train.StateHoliday = train.StateHoliday != '0'
test.StateHoliday = test.StateHoliday != '0'
weather = join_df(weather, state_names, "file", "StateName")
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'
expand_date(weather, "Date", drop=False)
expand_date(googletrend, "Date", drop=False)
expand_date(train, "Date", drop=False)
expand_date(test, "Date", drop=False)
trend_de = googletrend[googletrend.file == 'Rossmann_DE']
store = join_df(store, store_states, "Store")
# print(len(store[store.State.isnull()]))


df1 = join_df(train, store, "Store")
df2 = join_df(test, store, "Store")
df1 = join_df(df1, googletrend, ["State", "Year", "Week"])
df2 = join_df(df2, googletrend, ["State", "Year", "Week"])
df1 = df1.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
df2 = df2.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
df1 = join_df(df1, weather, ["State", "Date"])
df2 = join_df(df2, weather, ["State", "Date"])

for df in (df1, df2):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns:
                df.drop(c, inplace=True, axis=1)

for df in (df1, df2):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)

for df in (df1, df2):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days

for df in (df1, df2):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0

for df in (df1, df2):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24

for df in (df1, df2):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days

for df in (df1, df2):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()

df = prep_elapsed(train)
df1 = join_df(df1, df, ['Store', 'Date'])

df = prep_elapsed(test)
df2 = join_df(df2, df, ['Store', 'Date'])

df1 = df1[df1.Sales!=0]
df1.reset_index(inplace=True)
df2.reset_index(inplace=True)

df1.to_feather(f'{path}train_valid.feather')
df2.to_feather(f'{path}test.feather')

print(df1.head().T.head(40))
