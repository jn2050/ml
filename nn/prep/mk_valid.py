import numpy as np
import os

def mk_valid(_dir):
    path = os.environ.get('DATA_PATH') + '/' + _dir
    base = path + '/train/'
    g = np.asarray(glob.glob(base + '*.jpeg'))
    n = g.shape[0]
    print('Base folder:', base)
    print('Number of files in base folder:', n)

    msk = np.random.rand(n) <= 0.2
    g2 = np.asarray(g[msk])[:int(n*0.2)]

    dest = path + '/valid/'
    if os.path.exists(dest):
        print('Destination path already exists:', dest)
        print('Aborted')
        exit(0)

    os.makedirs(dest)
    for i in range(g2.shape[0]):
        name = g2[i][len(base):]
        os.rename(base + name, dest + name)
    print('Number of files moved:', g2.shape[0])