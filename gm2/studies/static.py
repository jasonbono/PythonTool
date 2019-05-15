import gm2
from gm2 import np, plt
from gm2.plotutil import plot_ts

runs = [3928]# 3830, 3832
yoke = 'G'
azi = 2

runs = [3930]# 3830, 3832
yoke = 'G'
azi = 3


runs = [3419, 3425+1]
runs = [3670, 3678+1]
runs = [3736, 3739+1]

runs = np.arange(3751+3, 3760+1) # 5h
yoke = 'G'
azi = 3


ip = gm2.Interpolation(runs)
#tr_mp3   = ip.loadTrMultipoles(n=3)
#tr_mpRaw = ip.loadTrMultipoles(freqs=ip.tr.freq)
ip.loadTrMultipoles()


gm2.plotutil.plotTrFieldMap(ip.tr.freq.mean(axis=0))
plt.show()

def compareMp(tr_mp):
    f = plt.figure(figsize=[gm2.plotutil.figsize()[0]*1.5, gm2.plotutil.figsize()[1]])
    ax1 = plt.subplot(3, 1, 1)
    plot_ts(ip.tr.time[:,8], (tr_mp[:,0] - ip.tr_mp[:,0])*gm2.HZ2PPB, '.', markersize=2, color=gm2.colors[0], label="Dipole")
    plt.title("Dipole")
    ax1.set_ylabel("[ppb]")
    plt.gca().set_xticklabels([])
    ax2 = plt.subplot(3, 1, 2)
    plot_ts(ip.tr.time[:,8], (tr_mp[:,1] - ip.tr_mp[:,1])/ip.tr_mp[:,1].mean()*100.0, '.', markersize=2, color=gm2.colors[1], label="Norm Quad")
    ax2.set_ylabel("[%]")
    plt.title("Norm Quad")
    plt.gca().set_xticklabels([])
    ax3 = plt.subplot(3, 1, 3)
    plot_ts(ip.tr.time[:,8], (tr_mp[:,2] - ip.tr_mp[:,2])/ip.tr_mp[:,2].mean()*100.0, '.', markersize=2, color=gm2.colors[2], label="Skew Quad")
    ax3.set_ylabel("[%]")
    plt.subplots_adjust(wspace=0.1)
    plt.title("Skew Quad")
    gm2.despine()
    #handles1, labels1 = ax1.get_legend_handles_labels()
    #handles2, labels2 = ax2.get_legend_handles_labels()
    #f.legend(markerscale=6)
    plt.show()

#compareMp(tr_mp3)
#compareMp(tr_mpRaw)


def tryMpOverview():
    means = [ip.tr_mp[:100, 0].mean(),
             ip.tr_mp[:100, 1].mean(),
             ip.tr_mp[:100, 2].mean()]


    f = plt.figure(figsize=[gm2.plotutil.figsize()[0]*1.5, gm2.plotutil.figsize()[1]])
    ax1 = f.subplots()
    plot_ts(ip.tr.time[:,8], ip.tr_mp[:,0] - means[0], '.', markersize=2, color=gm2.colors[0], label="Dipole    %.0f Hz" % means[0])
    ax2 = ax1.twinx()
    plot_ts(ip.tr.time[:,8], ip.tr_mp[:,1] - means[1], '.', markersize=2, color=gm2.colors[1], label="Norm Quad %.0f Hz/90mm" % means[1])
    plot_ts(ip.tr.time[:,8], ip.tr_mp[:,2] - means[2], '.', markersize=2, color=gm2.colors[2], label="Skew Quad %.0f Hz/90mm" % means[2])

    ax1.set_ylabel("Dipole change [Hz]")
    ax2.set_ylabel("Quad. change  [Hz/90mm]")
    plt.title("Trolley")
    gm2.despine()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    f.legend(handles1+handles2, labels1+labels2, markerscale=6)
    plt.show()

def tryProbeOverview(start=0, end=5):
    n = end-start
    f = plt.figure(figsize=[gm2.plotutil.figsize()[0]*1.50, gm2.plotutil.figsize()[1]*n/4.])
    for i, probe in enumerate(range(start, end)):
        print(n, i, probe)
        plt.subplot(n, 1, i+1)
        data_ = ip.tr.freq[:, probe] - ip.tr_mp[:, 0]
        plot_ts(ip.tr.time[:, probe], data_ - data_.mean(), '.', markersize=2, color=gm2.colors[i%len(gm2.colors)], label="probe %i" % (probe + 1))
        plot_ts(np.array([ip.tr.time[:, probe].min(), ip.tr.time[:, probe].max()]), [0.0, 0.0], ':', color='black', alpha=0.4, linewidth=1)
        plt.ylim([-6.0, 6.0])
        if i == n//2:
            plt.ylabel(r"$f_{\rm{trolley}}^{i} - f_{\rm{dipole}} - f_{\rm{mean}}$ [Hz]")
        if i != n-1:   
            plt.gca().set_xticklabels([])
        plt.title("probe %i: %.1f Hz" % ((probe+1), data_.mean()))
    #plt.legend()
    plt.subplots_adjust(wspace=0.1)
    gm2.despine()
    plt.show()


def tryOuterProbeOverview(start=5, end=17):
    pos = [12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11]
    n = end-start
    f = plt.figure(figsize=[gm2.plotutil.figsize()[0]*1.5, gm2.plotutil.figsize()[1]*n/4.])
    for i, probe in enumerate(range(start, end)):
        print(n, i, probe)
        plt.subplot(n, 2, pos[i])
        data_ = ip.tr.freq[:, probe] - ip.tr_mp[:, 0]
        plot_ts(ip.tr.time[:, probe], data_ - data_.mean(), '.', markersize=2, color=gm2.colors[i%len(gm2.colors)], label="probe %i" % (probe + 1))
        plot_ts(np.array([ip.tr.time[:, probe].min(), ip.tr.time[:, probe].max()]), [0.0, 0.0], ':', color='black', alpha=0.4, linewidth=1)
        plt.ylim([-6.0, 6.0])
        if pos[i] in [7]:
            plt.ylabel(r"$f_{\rm{trolley}}^{i} - f_{\rm{dipole}} - f_{\rm{mean}}$ [Hz]")
        if pos[i] not in [11, 12]:   
            plt.gca().set_xticklabels([])
        plt.title("probe %i: %.1f Hz" % ((probe+1), data_.mean()))
    #plt.legend()
    plt.subplots_adjust(wspace=0.1)
    gm2.despine()
    plt.show()


#tryMpOverview()
#tryProbeOverview()
#tryOuterProbeOverview()



#s = (ip.tr_time[:,0] > 1e18)&(ip.tr_freq[:,0]>52000)&(ip.tr_freq[:,0]<53000)



gm2.plotutil.plotTrFieldMap(gm2.util.getTrMultipole(ip.tr.freq.mean(axis=0), n=9) ) 

for mp in np.arange(1,10,2):
    mp_         = gm2.util.getTrMultipole(ip.tr.freq.mean(axis=0), n=mp)
    data_       = ip.fp.freq[:,ip.fp.select(yokes=[yoke], aziIds=[azi])].mean(axis=0)
    mp_rescale = [mp_[0]]
    for i in range(1,len(mp_)):
        mp_rescale.append(mp_[i] / 45.0**((i+1)//2))
    prediction_ = gm2.util.multipole((gm2.FP.probes.position.r,gm2.FP.probes.position.theta), *mp_rescale)
    plt.plot(prediction_ - data_,'.', label="n="+str(mp))

plt.xlabel("fixed probe")
plt.title("yoke "+yoke+" "+str(azi))
plt.ylabel(r"$f_{\rm{predicted}} - f_{\rm{fixedprobe}}$")
plt.gca().set_xticklabels([' ', 'TI', 'TM', 'TO', 'BI', 'BM', 'BO'])
plt.legend()
gm2.despine()
plt.show()

#ip.loadTrMultipoles(freqs=ip.tr.freq)

#tr_mpAt = []
#for mp in range(9):
#    tr_mpAt.append(gm2.util.interp1d(ip.tr.time[:,8], ip.tr_mp[s_tr,probe]))
#
#tr_mpAt = [gm2.util.interp1d()]

s_tr = (ip.tr.time[:,0] > 1e18)&(ip.tr.freq[:,0] > 52000)&(ip.tr.freq[:,0] < 53000)
tr_freq = []
for probe in range(17):
    tr_freq.append(gm2.util.interp1d(ip.tr.time[s_tr,probe], ip.tr.freq[s_tr,probe]))
skip = 100 # to make sure fp times are always within tr interpolation range
skipb = 2000


s = ip.fp.select(yokes=[yoke], aziIds=[azi], layers=['B'], radIds=['M'])
f_fp = ip.fp.freq[skip:-skipb, s].mean(axis=1)
f_tr = tr_freq[0](ip.fp.time[skip:-skipb, s].mean(axis=1))

from scipy import odr

def fp2tr_m(p, f):
    return (p[:,None]*f).mean(axis=0)
def fp2tr_mo(p, f):
    return (p[:-1,None]*f + p[-1]).mean(axis=0)
def fp2tr_p(p, f):
    return (p[:,None]+f).mean(axis=0)
    #return f.mean(axis=0)+p[0]
def fp2tr(p, f):
    return (p[:-p.shape[0]//2,None] * f + p[-p.shape[0]//2:,None]).mean(axis=0)
def fp2trf(p, f):
    n = f.shape[0]
    return (p[:n,None] + p[n:2*n,None] * (f - p[n*2:n*3,None]) ).mean(axis=0)
    #return (p[:n,None] * f + p[-p.shape[0]//2:,None]).mean(axis=0)
b0   = np.zeros([ip.fp.freq[:,s].shape[1]])
b0f  = np.zeros([ip.fp.freq[:,s].shape[1]*2])
b0f2 = np.zeros([ip.fp.freq[:,s].shape[1]*3])
b06  = np.zeros([ip.fp.freq[:,s].shape[1]+1])
x_ = ip.fp.freq[skip:-skipb, s]
y_ = tr_freq[0](ip.fp.time[skip:-skipb, s].mean(axis=1))

odr_m  = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2tr_m),  beta0=b0)
odr_mo = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2tr_mo), beta0=b06)
odr_p  = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2tr_p),  beta0=b0)
odr_   = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2tr),    beta0=b0f)
odr_f  = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2trf),   beta0=b0f2)
odrout_m  = odr_m.run()
odrout_mo = odr_mo.run()
odrout_p  = odr_p.run()
odrout_   = odr_ .run()
odrout_f  = odr_f.run()

plt.plot(fp2tr_m(odrout_m.beta, x_.T)-f_tr,  '.', alpha=0.5, markersize=2,  label=r"$f_i \cdot p_i$")
plt.plot(fp2tr_mo(odrout_mo.beta, x_.T)-f_tr,'.', alpha=0.5, markersize=2,  label=r"$f_i \cdot p_i + f_0$")
plt.plot(fp2tr_p(odrout_p.beta, x_.T)-f_tr,  '.', alpha=0.5, markersize=2,  label=r"$f_i + p_i$")
plt.plot(fp2tr(odrout_.beta, x_.T)-f_tr,     '.', alpha=0.5,  markersize=2, label=r"$f_i \cdot p_i + p_i$")
plt.plot(fp2trf(odrout_f.beta, x_.T)-f_tr,  '.', alpha=0.5,  markersize=2, label=r"$p_i * p_i(f_i - p_i)$")
#plt.plot(f_tr,'--')
#plt.plot(f_tr, f_fp-f_tr, '.')
#plt.plot(f_fp, f_fp-f_tr, '.')
plt.xlabel("fixed probe")
plt.ylabel("trolley")
gm2.despine()
plt.legend()
plt.show()


#s = ip.fp.select(yokes=[yoke, chr(ord(yoke)-1), chr(ord(yoke)+1)])


def fp2trs(p, f):
    return p[0] + p[1] * (f - p[2])
s = ip.fp.select(yokes=[yoke], aziIds=[azi])
w_p = []
for probe in range(5):
    print("probe ", str(probe))
    f_tr = tr_freq[probe](ip.fp.time[skip:-skipb, s].mean(axis=1))

    odrs  = []
    odrs_ = []
    for i in range(ip.fp.freq[skip:-skipb, s].shape[1]):
        print(i)
        xx_ = ip.fp.freq[skip:-skipb, s][:,i]
        y_ = tr_freq[probe](ip.fp.time[skip:-skipb, s].mean(axis=1))
        odrs.append(odr.ODR(odr.Data(xx_, y_), odr.Model(fp2trs),  beta0=[0., 1., 0.], ifixb=[1, 0, 0]))
        odrs_.append(odrs[i].run())
        #plt.plot(fp2trs(odrs_[i].beta, xx_)-f_tr,  '.', alpha=0.5, markersize=2, label="#"+str(i))
    #plt.show()


    def w(p, f):
        return(p[:,None]*f).sum(axis=0)

    b0  = np.zeros([ip.fp.freq[:,s].shape[1]])
    xn_ = np.empty_like(ip.fp.freq[skip:-skipb, s])
    for i, odr_ in enumerate(odrs_):
        xn_[:,i] = fp2trs(odr_.beta,  ip.fp.freq[skip:-skipb, s][:,i])
    y_ = tr_freq[probe](ip.fp.time[skip:-skipb, s].mean(axis=1))

    odr_sum  = odr.ODR(odr.Data(xn_.T, y_), odr.Model(w),  beta0=b0)
    odr_sum_ = odr_sum.run()

    #plt.plot(fp2trf(odrout_f.beta, x_.T)-f_tr,  '.', alpha=0.5,  markersize=2, label=r"$p_i * p_i(f_i - p_i)$")
    plt.plot(w(odr_sum_.beta, xn_.T)-f_tr,  '.', alpha=0.5,  markersize=2, label=r"")
    w_p.append(odr_sum_.beta)

gm2.despine()
plt.xlabel("time [sample]")
plt.ylabel(r"f_{\rm{model}} - f_{\rm{trolley}}")
plt.show()





import keras
import tensorflow as tf

x_ = ip.fp.freq[skip:-skipb, s]
probe = 0
y_ = tr_freq[probe](ip.fp.time[skip:-skipb, s].mean(axis=1))

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(6,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    #keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1)
    ])

optimizer = tf.train.RMSPropOptimizer(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
history = model.fit(x_, y_, epochs=300,
                    validation_split=0.2, verbose=1,
                    callbacks=[early_stop])

tt = model.predict(x_)
plt.plot(tt[:,0]-y_,'.', markersize=2)
#plt.plot(fp2trf(odrout_f.beta, x_.T)-y_,  '.', alpha=0.5,  markersize=2, label=r"$p_i * p_i(f_i - p_i)$")
#plt.plot(w(odr_sum_.beta, xn_.T)-y_,  '.', alpha=0.5,  markersize=2, label=r"")
plt.show()
