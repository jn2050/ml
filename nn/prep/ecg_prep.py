from xml.dom import minidom
import numpy as np
import os, re

path = os.path.join(os.environ.get('DATA_PATH'), 'ecg')
fname = os.path.join(path, 'ECG_CardioOn_200.XML')
xdoc = minidom.parse(fname)
root = xdoc.getElementsByTagName('ecgs')[0]
ecgs = xdoc.getElementsByTagName('ecg')
#print(ecgs)

for ecg in ecgs[:]:
    key = ecg.getElementsByTagName('id')[0].firstChild.data
    report = ecg.getElementsByTagName('report')[0].firstChild.data.replace('\n', ' ').replace(',', ' ')
    print(f'{key},{report}'); continue
    rhythms = ecg.getElementsByTagName('rhythm')[0]
    chdata = []
    for channel in rhythms.getElementsByTagName('channel'):
        data =channel.firstChild.data.split()
        data = np.array(data).astype(float)
        chdata.append(data)
    chdata = np.vstack(chdata)
