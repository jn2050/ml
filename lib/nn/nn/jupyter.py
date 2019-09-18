import pandas as pd
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML


def widget(label, x, f='N'):
    label += ' '
    if f=='C':
        x = f'{x:,.0f}'.replace(',', '_').replace('.', ',').replace('_', '.')
        u = '€'
    if f=='P':
        x = f'{100*x:,.1f}'.replace(',', '_').replace('.', ',').replace('_', '.')
        u = '%'
    s = '<div class=\"number-box\">' \
            + '<div class=\"number-label\">'+ label  + '</div>' \
            + '<div class=\"number-unit\">'+ u  + '</div>' \
            + '<div class=\"number-value\">' + x + '</div>' \
            + '</div>'
    display(HTML(s))


def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)


def table(data, cols=None, title=None):
    if title is not None: print(title)
    df1 = pd.DataFrame(data, columns=cols)
    return df1


def meta(df, order=False):
    data = []
    for c in df.columns:
        unq = len(df[c].cat.categories) if df[c].dtype.name == 'category' else len(df[c].unique())
        examples = '; '.join([str(e) for e in df[c].value_counts().index[:5]]) 
        data.append((c, df[c].dtypes, unq, unq/len(df), df[c].isna().sum(), df[c].isna().sum()/len(df), examples))
    df1 = pd.DataFrame(data, columns=['attr', 'type', 'uniques', 'uniques_p', 'nulls', 'nulls_p', 'examples'])
    if order: df1.sort_values(['uniques'], ascending=False, inplace=True)
    return df1


def meta_to_excel(df, fname):
    writer = pd.ExcelWriter(fname)
    df_meta = meta(df, order=False)
    df_meta.to_excel(writer,'Original order')
    df_meta = meta(df, order=True)
    df_meta.to_excel(writer,'Uniques order')
    writer.save()


def calc_hist(df, col, n, n2):
    x = df[col].value_counts().values
    y = list(df[col].value_counts().index.values)
    if n is None:
        total, t = sum(x), 0
        for k,e in enumerate(x):
            t += e
            if (n2 is None and t>=total*0.99) or (n2 is not None and k>=n2-1):
                break
        k += 1
        n = k if len(df[col].value_counts())>k else len(df[col].value_counts())
    if n<len(df[col].value_counts()):
        r = sum(x[n:])
        x, y = x[:n], y[:n]
        x, y = np.append(x, [r]), np.append(y, ['Etc'])
        n += 1
    return x, y, n


def format_chart_titles(ax, title=None, xlabel=None, ylabel=None):
    sns.set(font_scale=1)
    title = '' if title is None else title    
    xlabel = '' if xlabel is None else xlabel
    ylabel = '' if ylabel is None else ylabel
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12, rotation='vertical')
    ax.set_ylabel(ylabel, fontsize=12, rotation='vertical')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'.replace(',', '.')))
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'.replace(',', '.')))

        
def format_chart(ax, title, ylabel, wtotal, max_x, w):
    format_chart_titles(ax, title, '', ylabel)
    for p in ax.patches:
        p.set_height(1)
        width = p.get_width()
        ax.text(width+max_x*0.02*(15/w), p.get_y()+p.get_height()*0.7, '{:1.2f}'.format(width/wtotal), ha='center')
    plt.subplots_adjust(hspace = 1.0)


def plot_hist_cat(df, col, n=None, n2=20, w=15):
    x, y, n_ = calc_hist(df, col, n, n2)
    h = n_/2.0-0.3
    plot = plt.figure(figsize=(w, h))
    ax = sns.barplot(x=x, y=y, orient='h')
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    format_chart(ax, col, '#', float(len(df)), max(x), w)
    return plot


def plot_hist_cat_grid(df, cols, n=None, n2=20, grid_cols=2, w=15, h1=1.0):
    grid_rows = len(cols)//grid_cols
    lines = []
    for i,col in enumerate(cols):
        x, y, n_ = calc_hist(df, col, n, n2)
        lines.append(n_)
    avg = sum(lines)/len(lines)
    h = grid_rows*avg*w/(20/h1)
    #if h>15*grid_rows: h = 15*grid_rows
    if h>0.4*sum(lines): h = 0.4*sum(lines)
    plot = plt.figure(figsize=(w, h))
    for i,col in enumerate(cols):
        x, y, _ = calc_hist(df, col, n, n2)
        plt.subplot(grid_rows+1,grid_cols,i+1)
        ax = sns.barplot(y=y, x=x, orient='h')
        plt.gca().set_facecolor((1.0, 1.0, 1.0))
        format_chart(ax, col, '#', float(len(df)), max(x)*grid_cols, w)
    plt.tight_layout()
    return plot


def plot_hist_int(df, col, w=15, h=None):
    df = df.copy()
    df[col] = df[col].fillna(0).astype(np.int)
    #if h is None: h = w/4
    h = w/4 if h is None else h
    plot = plt.figure(figsize=(w, h))
    ax = sns.countplot(df[col])
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    t1, t2 = df[col].min(), df[col].max()
    if t2-t1>50:
        r = np.arange(t1, t2, step=max((t2-t1)//50,2))
        plt.xticks(r, r)
    for p in ax.patches:
        p.set_width(1)
    format_chart_titles(ax, col, '', '#')
    ax.set_title(col, fontsize=14)
    ax.set_ylabel('#', fontsize=14, rotation='vertical')
    return plot


def table_hist(df, col):
    x, y, _ = calc_hist(df, col, 20, 20)
    df1 = pd.DataFrame({'Value': y, '#': x })
    n = sum(df1['#'])
    df1['%'] = df1['#']/n
    return df1


def plot_dist(df, col, _max=None, w=15):
    #np.array_equal(df['idade'].fillna(0), df['idade'].fillna(0).astypde(int))
    df = df.copy()
    df = df[df[col].isna()==False]
    if _max is not None:
        df.loc[df[col]>_max, col] = _max
    plot = plt.figure(figsize=(w, w/3))
    ax = sns.distplot(df[col], kde=True)
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    format_chart_titles(ax, col, '', 'p')
    return plot


def plot_bins(df, col, bins, w=15):
    df = df.copy()
    df[col] = pd.cut(df[col], bins, right=False)
    plot = plt.figure(figsize=(w, w/3))
    ax = sns.countplot(df[col])
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    ax.set_title(col, fontsize=14)
    ax.set_xlabel('', fontsize=14, rotation='vertical')
    ax.set_ylabel('#', fontsize=14, rotation='vertical')
    total = len(df)
    for p in ax.patches:
        p.set_width(1)
        height = p.get_height()
        #ax.text(p.get_x()+p.get_width()*0.5, p.get_y()+p.get_height()*1.05, '{:1.2f}'.format(height/total), ha='center')
    return plot


def plot_bar(df, col1, col2, w=10, h=None):
    h = w/3 if h is None else h
    plot = plt.figure(figsize=(w, h))
    ax = sns.barplot(x=col1, y=col2, data=df)
    format_chart_titles(ax, '', '', col2)
    for p in ax.patches:
        p.set_width(1)
    return plot


def plot_bar_sum(df, col1, col2, w=10):
    df1 = df.copy().groupby([col1])[col2].sum().reset_index()
    df1.columns = [col1, col2]
    return plot_bar(df1, col1, col2, w)


def build_df_corr_cat_cat(df, col1, col2, nmin=100):
    df = df[df[col1].isna()==False]
    data = []
    for v1 in df[col1].unique():
        for v2 in df[col2].unique():
            n = len(df[df[col1]==v1])
            n2 = len(df[(df[col1]==v1)&(df[col2]==v2)])
            v = n2/n
            data.append((v1, v2, n, n2, v))
    df = pd.DataFrame(data, columns=[col1, col2, 'n', 'n2', 'v'])
    return df


def plot_corr_cat_cat(df, col1, col2):
    left = None
    for dep in sorted(list(df[col2].unique()), reverse=False):
        df1 = df[df[col2]==dep]
        if dep==0: df1 = df1.sort_values(['v'], ascending=True)
        else: df1 = df1.sort_values(['v'], ascending=False)
        x, y, sizes = df1[col1].values, df1['v'].values, df1['n2'].values
        if left is None: left = np.zeros(len(x))
        ax = plt.barh(x, y, 1.0, left=left)
        plt.gca().set_title(f'{col1.capitalize()} / {col2.capitalize()}', fontsize=12)
        for i,p in enumerate(ax.patches):
            k = len(ax.patches)
            i2 = k-i-1
            ypos = i2*1 + 0.2
            if dep==0:
                s1 = f'{100*y[i2]:.1f}%'.replace(',', '.')
                s2 = f'(N={sizes[i2]:,})'
                plt.annotate(s1, (0.010, ypos), color='white', fontsize=10, fontweight='bold', horizontalalignment='left')
                plt.annotate(s2, (0.1, ypos), color='white', fontsize=10, fontweight='bold', horizontalalignment='left')
            else:
                s = f'{100*y[i2]:.1f}%'.replace(',', '.')
                plt.annotate(s, (0.995, ypos), color='white', fontsize=10, fontweight='bold', horizontalalignment='right')
        left += y
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    plt.gca().margins(0.0)
    plt.gca().invert_yaxis()
    return ax


def build_df_corr_cat_cont(df, col1, col2, nmin=100, skip_zero=True):
    df = df[df[col1].isna()==False]
    data = []
    for v1 in df[col1].unique():
        df1 = df[df[col1]==v1]
        if len(df1)<nmin: continue
        if skip_zero: df1 = df1[df1[col2]>0]
        n = len(df1)
        v = sum(df1[col2])/n if n>0 else 0
        data.append((v1, n, v))            
    df = pd.DataFrame(data, columns=[col1, 'n', 'v']).sort_values(['v'], ascending=False)
    return df


def plot_corr_cat_cont(df, col1, col2):
    x, y, sizes = df[col1].values, df['v'].values, df['n'].values
    ax = plt.barh(x, y, 1.0)
    plt.gca().set_title(f'{col1.capitalize()} / {col2.capitalize()}', fontsize=12)
    for i,p in enumerate(ax.patches):
        i2 = len(ax.patches)-i-1
        ypos = i2*1 + 0.2
        pad = max(y)*0.01
        s1 = f'(N={sizes[i2]:,})'.replace(',', '.')
        s2 = f'{y[i2]:,.0f}'.replace(',', '.')
        plt.annotate(s1, (pad, ypos), color='white', fontsize=10, fontweight='bold', horizontalalignment='left')
        plt.annotate(s2, (y[i2]-pad, ypos), color='white', fontsize=10, fontweight='bold', horizontalalignment='right')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'.replace(',', '.')))
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    plt.gca().margins(0.0)
    plt.gca().invert_yaxis()
    return ax


def plot_corr_cat(df, col1, col2, nmin=100, skip_zero=True, w=15, plot=None):
    if df[col2].dtype == 'int64':
        df = build_df_corr_cat_cat(df, col1, col2, nmin)
        if plot is None: plot = plt.figure(figsize=(w, len(df)/2*0.4))
        return plot_corr_cat_cat(df, col1, col2)
    else:
        df = build_df_corr_cat_cont(df, col1, col2, nmin, skip_zero)
        if plot is None: plot = plt.figure(figsize=(w, len(df)*0.4))
        return plot_corr_cat_cont(df, col1, col2)


def plot_corr_cat_2(df, col1, col2, col3, w=15):
    df1 = build_df_corr_cat_cat(df, col1, col2)
    df2 = build_df_corr_cat_cont(df, col1, col3)
    plot = plt.figure(figsize=(w, len(df2)*0.4))
    ax = plt.subplot(1,2,1)
    plot_corr_cat_cat(df1, col1, col2)
    ax = plt.subplot(1,2,2)
    plot_corr_cat_cont(df2, col1, col3)
    return plot


def plot_corr_cont_cat(df, col1, col2, w=10):
    plot = plt.figure(figsize=(w, w/3))
    df1 = df[df[col1].isna()==False]
    for v2 in df1[col2].unique():
        df2 = df1[df1[col2]==v2]
        ax = sns.distplot(df2[col1], kde=True, hist=False)
    plt.gca().set_facecolor((1.0, 1.0, 1.0))
    return plot


def plot_corr_cont_cont(df, col1, col2, col3, w=15):
    plot = plt.figure(figsize=(w, w/3))
    df1 = df[df[col1].isna()==False]
    plot = plt.figure(figsize=(w, w/3))
    #sns.jointplot(x='idade', y='D', data=df)
    sns.scatterplot(x='idade', y='D', data=df)
    #sns.lmplot(x='idade', y='D', data=df)
    return plot


def plot_corr_cont(df, col1, col2, skip_zero=True, w=10):
    if df[col2].dtype == 'int64':
        return plot_corr_cont_cat(df, col1, col2, w)
    else:
        return plot_corr_cont_cont(df, col1, col2, skip_zero, w)

