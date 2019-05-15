import gm2
from gm2 import np, plt

runs = [3928]# 3830, 3832
yoke = 'G'
azi = 2

runs = [3930]# 3830, 3832
yoke = 'G'
azi = 3

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3

ip = gm2.Interpolation(runs)
#ip.loadFpRes()


s_tr = (ip.tr_time[:,0] > 1e18)&(ip.tr_freq[:,0] > 52000)&(ip.tr_freq[:,0] < 53000)
tr_dipole = gm2.util.interp1d(ip.tr_time[s_tr,0], ip.tr_freq[s_tr,0])
skip = 10 # to make sure fp times are always within tr interpolation range
station_phi = (np.arctan2(ip.fp.getY(), ip.fp.getX())[ip.fp.select(yokes=[yoke], aziIds=[azi])][0]/np.pi*180 + 360) % 360.


figsize = figsize=gm2.plotutil.figsize()
figsize = [figsize[0]*1.5, figsize[1]*1.0]
image_names = []


plt.figure(figsize=figsize)
for yoke_ in np.arange(ord('A'), ord('A')+12):
    for azi_ in np.arange(1, 7):
        print(chr(yoke_)+str(azi_))
#for yoke_ in [ord(yoke)]:
#    for azi_ in [azi-1]:
        station_phi_ = (np.arctan2(ip.fp.getY(), ip.fp.getX())[ip.fp.select(yokes=[chr(yoke_)], aziIds=[azi_])][0]/np.pi*180 + 360) % 360.
        i = 0
        for layer in ['T', 'B']:
            for rad in ['O', 'M', 'I']:
                i += 1
                plt.subplot(2, 3, i)
                s = ip.fp.select(yokes=[chr(yoke_)], aziIds=[azi_], layers=[layer], radIds=[rad])
                if np.argwhere(s).shape[0] > 0:
                    t0 = ip.fp_time[skip:-skip, s].min()
                    trFpDif_ = ip.fp_freq[skip:-skip, s] - tr_dipole(ip.fp_time[skip:-skip, s])
                    mean_ = trFpDif_.mean()
                    plt.plot((ip.fp_time[skip:-skip, s]-t0)/1e9/3600, trFpDif_-mean_, '.', markersize=2)
                    plt.ylim([-50, +50])
                plt.title(layer+rad)
                if i in [2]:
                    plt.title("Station "+chr(yoke_)+str(azi_)+" ("+r"$\Delta\phi$ "+("%.1f deg)"  % (station_phi_-station_phi))+"\n"+layer+rad)
                if i != 5:
                    plt.gca().set_xticklabels([])
                else:
                    plt.xlabel("time [h]")
                if i in [1, 4]:
                    plt.ylabel(r"$f_{\rm{fp}} - f_{\rm{trly}}$ [Hz]")
                else:
                    plt.gca().set_yticklabels([])
        gm2.despine()
        fname = "plots/static_"+str(runs[0])+"_"+chr(yoke_)+str(azi_)+".png"
        plt.savefig(fname)
        image_names.append(fname)
        plt.clf()
        #plt.show()

import imageio
images = []
for iname in image_names:
    images.append(imageio.imread(iname))
imageio.mimsave("plots/stativ_"+str(runs[0])+".gif", images, format='GIF', duration=1)
