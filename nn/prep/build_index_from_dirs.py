"""
    Builds dict from dirs
    Files are arranged in folder classes

"""

import pandas as pd
import os


def is_valid_file(fname):
    fname = fname.lower()
    return any(fname.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.npy'])


def find_classes(dsdir):
    classes = [d for d in os.listdir(dsdir) if os.path.isdir(os.path.join(dsdir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(basepath, dsdir):
    rows = []
    for target in sorted(os.listdir(os.path.join(basepath, dsdir))):
        d = os.path.join(basepath, dsdir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                _, classdir = os.path.split(root)
                if is_valid_file(fname):
                    filename = os.path.join(dsdir, classdir, fname)
                    rawfilename = os.path.join(dsdir, classdir, fname)
                    row = (filename, rawfilename, target)
                    rows.append(row)
    return rows


def build_dict_from_dirs(path):
    classes, class_to_idx = find_classes(os.path.join(path, 'train'))
    print('Found {:d} Classes:'.format(len(classes)), classes)

    rows = make_dataset(path, 'train')
    print('Processed {:d} files in train folder'.format(len(rows)))
    df = pd.DataFrame.from_records(rows, columns=['filename', 'rawfilename', 'class'])
    df.to_csv(os.path.join(path, 'train.csv'), index=False)

    rows = make_dataset(path, 'valid')
    print('Processed {:d} files in valid folder\n'.format(len(rows)))
    df = pd.DataFrame.from_records(rows, columns=['filename', 'rawfilename', 'class'])
    df.to_csv(os.path.join(path, 'valid.csv'), index=False)


path = os.path.join(os.environ.get('DATA_PATH'), 'dogs')
build_dict_from_dirs(path)

