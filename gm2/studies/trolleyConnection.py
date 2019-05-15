import gm2
from gm2 import plt, np
runs = [3997]

bc = gm2.Barcode(runs)
tr = gm2.Trolley(runs)
tr_time, tr_phi, tr_freq = tr.getBasics()

timeAtPos = gm2.util.interp1d(tr_phi[:,0], tr_time[:,0])
posAtTime = gm2.util.interp1d(tr_time[:,0], tr_phi[:,0])

start_phi  = tr_phi[1,0]-0.001
start_time = timeAtPos(start_phi)
dt = timeAtPos(start_phi - 2*np.pi) - start_time



window = 30  # in sec
s1 = (bc.time >= start_time) & (bc.time < start_time + window * 1e9)
plt.plot(posAtTime(bc.time[s1]), bc.abs[0][s1],   '.', markersize=1, label="downstream (abs 1)")
plt.plot(posAtTime(bc.time[s1]), bc.abs[1][s1]-1, '.', markersize=1, label="downstream (abs 2)")


s2 = (bc.time >= start_time+dt - window*1e9) & (bc.time < start_time + dt + window * 1e9) & (bc.time < tr_time[:,0].max())

plt.plot(posAtTime(bc.time[s2])+2*np.pi, bc.abs[0][s2]-2, '.', markersize=1, label="upstream (abs 1)")
plt.plot(posAtTime(bc.time[s2])+2*np.pi, bc.abs[1][s2]-3, '.', markersize=1, label="upstream (abs 2)")

#plt.plot(posAtTime(bc.time[s2])+2*np.pi, bc.abs[0][s2])
#plt.plot((bc.time[s2]-start_time - dt)/1e9, bc.abs[0][s2])

plt.xlabel("azimuth [deg]")
plt.ylabel("barcode [a.u.]")
plt.title("run "+str(runs[0]))
#plt.legend()

lgnd = plt.legend()

#change the marker size manually for both lines
for legendHandle in lgnd.legendHandles:
    legendHandle._legmarker.set_markersize(6)

gm2.despine()
plt.show()
