import gm2
from gm2 import plt, np

#bc = barcode([3997])
bc = gm2.Barcode([4058], cw=False)
#bc = gm2.Barcode([4059], cw=True)

#bc = gm2.Barcode([9999], cw=True)
# load file from different directory
#bc.tr.basedir = "/Volumes/D/Offline/"
#bc.tr.loadFiles([5052])
bc.load()


bc.loadPos(interpolate=2)

'''
tr_time, tr_phi, _ = bc.tr.getBasics()
t0 = bc.pos_time[0].min()
pp = gm2.util.interp1d(tr_time[10:,0], tr_phi[10:,0])
data = [[bc.pos_no[0]*62.,  gm2.R * pp(bc.pos_time[0])],
        [bc.pos_no[1]*62.,  gm2.R * pp(bc.pos_time[1])]]

#ax = plt.subplot(211)
#plt.plot((bc.pos_time[0]-t0)/1e9, data[0][1]-data[0][1].mean(), '.', markersize=2, label="new bc")
#plt.plot((bc.pos_time[0]-t0)/1e9, data[0][0]-data[0][0].mean(), '.', markersize=2, label="ref bc")
plt.plot((data[0][0]-data[0][0].mean())/gm2.R/np.pi*180 + 180, (data[0][0]-data[0][0].mean() - data[0][1]-data[0][1].mean())/gm2.R/np.pi*180, '.', markersize=2, label="ref bc")

#plt.gca().set_xticklabels([])
plt.setp( ax.get_xticklabels(), visible=False)
#plt.legend()
plt.ylabel("position difference [deg]")

#plt.subplot(212, sharex=ax)
#plt.plot((bc.pos_time[0]-t0)/1e9, (data[0][1]-data[0][0]) - (data[0][1]-data[0][0]).mean(), '.', markersize=2, label="encoder 1")
#plt.plot((bc.pos_time[1]-t0)/1e9, (data[1][1]-data[1][0]) - (data[1][1]-data[1][0]).mean(), '.', markersize=2, label="encoder 2")
#plt.xlabel("time [s]")
plt.xlabel("azimuth [deg]")
#plt.legend()
plt.title("Barcode Vs Encoder")
gm2.sns.despine()
plt.show()
'''

tr_time, tr_phi, _ = bc.tr.getBasics()
t0 = bc.pos_time[0].min()
pp = gm2.util.interp1d(tr_time[100:,0], tr_phi[100:,0])
data = [[bc.pos_no[0]*60.,  np.pi/180 * gm2.R * pp(bc.pos_time[0])],
        [bc.pos_no[1]*60.,  np.pi/180 * gm2.R * pp(bc.pos_time[1])]]

ax = plt.subplot(211)
plt.plot((bc.pos_time[0]-t0)/1e9, data[0][1]-data[0][1].mean(), '.', markersize=2, label="new bc")
plt.plot((bc.pos_time[0]-t0)/1e9, data[0][0]-data[0][0].mean(), '.', markersize=2, label="ref bc")
#plt.gca().set_xticklabels([])
plt.setp( ax.get_xticklabels(), visible=False)
plt.legend()
plt.ylabel("barcode position [mm]")

plt.subplot(212, sharex=ax)
plt.plot((bc.pos_time[0]-t0)/1e9, (data[0][1]-data[0][0]) - (data[0][1]-data[0][0]).mean(), '.', markersize=2, label="encoder 1")
plt.plot((bc.pos_time[1]-t0)/1e9, (data[1][1]-data[1][0]) - (data[1][1]-data[1][0]).mean(), '.', markersize=2, label="encoder 2")
plt.xlabel("time [s]")
plt.ylabel("difference \n barcode position [mm]")
plt.legend()
gm2.sns.despine()
plt.show()


s_tr = ((tr_time[:,0]   > t0+250e9) & (tr_time[:,0]   < t0+400e9))
s_bc = ((bc.pos_time[0] > t0+250e9) & (bc.pos_time[0] < t0+400e9))

f_bc = gm2.util.fit_lin(bc.pos_time[0][s_bc],   bc.pos_no[0][s_bc]*60.)
f_tr = gm2.util.fit_lin(tr_time[:,0][s_tr],    tr_phi[:,0][s_tr]* np.pi/180. * gm2.R)

#plt.plot(bc.val_times[0][s_bc], bc.val_num[0][s_bc]*60., '.', markersize=2, color=gm2.colors[0])
#plt.plot(bc.val_times[0][s_bc], gm2.util.func_lin(bc.val_times[0][s_bc], *f_bc), '-', color=gm2.colors[0])
#plt.show()


plt.plot((bc.pos_time[0][s_bc]-t0)/1e9, (data[0][1][s_bc]-data[0][0][s_bc]) - (data[0][1][s_bc]-data[0][0][s_bc]).mean(), '.', markersize=2, label="difference")
plt.plot((tr_time[:,0][s_tr]-t0)/1e9, gm2.util.func_lin(tr_time[:,0][s_tr], *f_tr)-tr_phi[:,0][s_tr]* np.pi/180. * gm2.R, '.',                label="residuals new bc")
plt.plot((bc.pos_time[0][s_bc]-t0)/1e9, bc.pos_no[0][s_bc]*60.-gm2.util.func_lin(bc.pos_time[0][s_bc], *f_bc), '.', markersize=2,          label="residuals ref bc")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("position difference [mm]")
gm2.despine()
plt.show()

#plt.plot(tr_time[:,0][s_tr], gm2.util.func_lin(tr_time[:,0][s_tr], *f_tr)-tr_phi[:,0][s_tr]* np.pi/180. * gm2.R, '.-', color=gm2.colors[0])
#plt.show()

#plt.plot(tr_time[:,0][s_tr], tr_phi[:,0][s_tr]* np.pi/180. * gm2.R, '.', markersize=2, color=gm2.colors[0])
#plt.plot(tr_time[:,0][s_tr], gm2.util.func_lin(tr_time[:,0][s_tr], *f_tr), '-', color=gm2.colors[0])
#
#plt.plot(tr_time[:,0][s_tr], gm2.util.func_lin(tr_time[:,0][s_tr], *f_tr)-tr_phi[:,0][s_tr]* np.pi/180. * gm2.R, '.', color=gm2.colors[0])
#plt.show()

'''
pp2 = gm2.util.interp1d(bc.val_times[0], bc.val_num[0])
tt = np.arange(t0+250e9, t0+400e9, 0.02*1e9)
tt_data  = pp(tt) * np.pi/180. * gm2.R
tt_data2 = pp2(tt) * 60.
sp = np.fft.fft(tt_data)
sp2 = np.fft.fft(tt_data2)
sp3 = np.fft.fft(tt_data - tt_data2)
freq = np.fft.fftfreq(tt.shape[-1]) * 1/0.2

plt.plot(tt, tt_data-tt_data.mean())
plt.plot(tt, tt_data2-tt_data2.mean())
plt.plot(tt, (tt_data-tt_data2)-(tt_data-tt_data2).mean())
plt.show()

plt.semilogy(freq, abs(sp), '.')
plt.semilogy(freq, abs(sp2), '.')
plt.semilogy(freq, abs(sp3), '.')
plt.show()
'''
