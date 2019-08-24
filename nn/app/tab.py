import pandas as pd
from lib.data import *
from lib.models import *
from lib.learner import Learner
from lib.metrics import *


def iris(cfg):
    cat_vars = []
    cont_vars = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
    dep_var = 'class'

    df = pd.read_csv(os.path.join(cfg.path, 'iris.csv'))
    val_idx = df_split(df, v=0.2)
    data = StructData(df, val_idx, cat_vars, cont_vars, dep_var, bs=256, reg=False, shuffle=True)
    model = StructNet(data, 1, [32,32,32], 0.00, [0.5,0.5,0.5], reg=False)
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.NLLLoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[accuracy], maximize=True, name='iris')
    learner.fit(cfg.epochs)

    y, y2, _ = learner.epoch(0)
    for i in range(y.shape[0]):
         print('y={:d}, y2=[{:2f},{:.2f},{:.2f}] -> {:d}'.format(y[i], y2[i][0], y2[i][1], y2[i][2], y2[i].max(0)[1][0]))
    learner.close()


def houses(cfg):
    cat_vars = []
    cont_vars = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    dep_var = 'MEDV'

    df = pd.read_csv(os.path.join(cfg.path, 'houses.csv'))
    val_idx = df_split(df, v=0.2)
    data = StructData(df, val_idx, cat_vars, cont_vars, dep_var, bs=256)
    model = StructNet(data, 1, [1024,512], 0.00, [0.7,0.7])
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.MSELoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[exp_percent], maximize=False, name='houses')
    learner.fit(cfg.epochs)

    y, y2, _ = learner.epoch(0)
    y = y.exp()
    y2 = y2.exp()
    error_percent = (y2 - y).abs() / y
    mse = (y - y2) ** 2
    error = (y2 - y) / y
    for i in range(y.shape[0]):
         print('{:.2f} {:.2f}  -> {:.2f} {:.1f}%'.format(y[i], y2[i], mse[i], 100*error[i]))
    print('Max: {:.1f}% Mean: {:.1f}% Std: {:.1f}%'.format(100*error_percent.max(), 100*error_percent.mean(), 100*error_percent.std()))
    learner.close()


def ross(cfg):
    cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen', 'Promo2Weeks',
                'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State',
                'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw', 'SchoolHoliday_fw',
                'SchoolHoliday_bw']
    cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity',
                   'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend',
                   'trend_DE', 'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']
    dep_var = 'Sales'

    df = pd.read_feather(f'{cfg.path}/train_valid.feather')
    #df = df_sample(df, 100000)
    val_idx = df_split_by_date(df, '2014-08-01', '2014-09-17')   # MODIFIED
    df = df[cat_vars + cont_vars + [dep_var]]

    data = StructData(df, val_idx, cat_vars, cont_vars, dep_var, bs=256)
    model = StructNet(data, 1, [1000,500], 0.04, [0.001,0.01])
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.MSELoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[exp_rmspe], maximize=False, name='ross')
    learner.fit(cfg.epochs)
    learner.close()

    # if cfg.predict is not None:
    #     print('Predict:', cfg.predict)
    #     df2 = df_sample(df, 5)
    #     y2 = learner.predict(df2)
    #     print(y2)
    #     exit(0)


def tab(cfg):
    ross(cfg)
    exit(0)
