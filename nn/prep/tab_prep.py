import pandas as pd
import os


def prep_iris(path, fname1, fname2):
    cols = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'class']
    df = pd.read_csv(os.path.join(path, fname1), header=None, names=cols)
    df = df.sample(frac=1)
    df.to_csv(os.path.join(path, fname2), index=False)


def prep_houses(path, fname1, fname2):
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv(os.path.join(path, fname1), delim_whitespace=True, header=None, names=cols)
    df = df.sample(frac=1)
    df.to_csv(os.path.join(path, fname2), index=False)


path = os.path.join(os.environ.get('DATA_PATH'), 'tab')
#prep_houses(path, 'houses_original.csv', 'houses.csv')
prep_iris(path, 'iris_original.csv', 'iris.csv')
