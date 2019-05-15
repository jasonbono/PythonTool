import FixedProbe
from util import *

import numpy as np


runs = [3997, 3998]
fp = [FixedProbe.FixedProbe([r]) for r in runs]
def callback(i):
    return [fp[i].getFrequencys(True), fp[i].getTimesGPS(), fp[i].getPhis()]

fp_freq = []
fp_time = []
fp_phi  = []
index   = []
for i, r in enumerate(runs):
    fp_freq.append([])
    fp_time.append([])
    fp_phi.append([])
    fp_freq[i], fp_time[i], fp_phi[i] = fp[i].loop(callback, i)


    nprobes = fp_freq[0].shape[1]
    index.append(np.zeros([nprobes]))
    for probe in range(nprobes):
        freq_ = fp_freq[i][:,probe,0]
        index[i][probe] = firstIndexAboveThr(freq_, freq_.mean() + 10 * freq_.std(),100)

nprobes = fp_freq[0].shape[1]
dt = np.zeros([nprobes])
df = np.zeros([nprobes])
offset = 20
for probe in range(nprobes):
    if (index[0][probe]-offset > 0)&(index[1][probe]-offset > 0):
        dt[probe] = fp_time[0][int(index[0][probe])-offset, probe]    - fp_time[1][int(index[1][probe])-offset, probe]
        df[probe] = fp_freq[0][int(index[0][probe])-offset, probe, 0] - fp_freq[1][int(index[1][probe])-offset, probe, 0]
    else:
        dt[probe] = np.nan
        df[probe] = np.nan

import matplotlib.pyplot as plt
plt.plot(-dt/1e9/60,df,'.')
plt.xlabel("difference time [min]")
plt.ylabel("difference frequency [Hz]")
plt.show()


import matplotlib.pyplot as plt
probe = fp_phi[0][0]==132.
t0 = fp_time[0][fp_time>0].min()

plt.plot((fp_time[0][:,probe]-t0)/1e9/60, fp_freq[0][:, probe, 0], '.')
plt.show()



