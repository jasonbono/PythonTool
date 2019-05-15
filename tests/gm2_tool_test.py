
#load the libraries
import gm2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np




##Trolley
#tr = gm2.Trolley([6327])
#tr_time, tr_pos, tr_freq = tr.getBasics() # return numpy arrays with size [n_events, n_probes]
#probe = 4
#plt.plot(tr_pos[:,probe], tr_freq[:,probe], 'o-', lw=2,alpha=0.5)
## plt.xlim(20000,80000)
#plt.title('Trolley Readings')
#plt.xlabel('Position (phi)')
#plt.ylabel('Frequency (Hz)')
#plt.grid()
#plt.tight_layout()
#plt.show()





#Fixed probes
freq_method = 2
evt = 5
probe = 4
fp = gm2.FixedProbe([6327])
fp.load(evt)
fp_freq = fp.getFrequency(freq_method)
fp_time = fp.getTimeGPS()

print(fp.data.Header_MuxId)
print(np.shape(fp_freq))
print(np.shape(fp_time))

plt.plot(fp_freq[probe], fp_time[probe], 'o-', lw=2,alpha=0.5)
# plt.xlim(20000,80000)
plt.title('Fixed Probe {} Readings'.format(probe))
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.tight_layout()
plt.show()


fp_time, fp_freq = fp.getBasics() # return numpy array with size [n_events, n_probes]
