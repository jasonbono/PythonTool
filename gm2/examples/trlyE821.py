# Example how to load E821 trolley data
# Data: https://gm2-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=3354
# Comments, 
# - not all trolley probes have the same number of measuerements
#   this forces a different handling then E989 data
# - the mapping from encoder to azimuthal position is my best guess
# - 

import gm2
from gm2 import plt
import os

path = os.environ['ARTTFSDIR']+"/../../E821/" # adjust!
fname = "trolleyData_21_3_01.txt"
trly = gm2.E821(path+fname)

for probe in [0,4,12]:
    plt.plot(trly.getPhi(probe)/gm2.np.pi*180., trly.getFrequency(probe)/1e3, '.', label="probe #"+str(probe))
plt.xlabel("azimuth [degree]")
plt.ylabel("frequency [kHz]")
plt.legend()
gm2.despine()
plt.title("E821 "+fname[:-4])
plt.show()
