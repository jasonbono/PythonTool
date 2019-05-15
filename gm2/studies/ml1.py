import gm2
from gm2 import np, plt

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
#odr_f  = odr.ODR(odr.Data(x_.T, y_), odr.Model(fp2trf),   beta0=b0f2)
odrout_m  = odr_m.run()
odrout_mo = odr_mo.run()
odrout_p  = odr_p.run()
odrout_   = odr_ .run()
#odrout_f  = odr_f.run()

plt.plot(fp2tr_m(odrout_m.beta, x_.T)-f_tr,  '.', alpha=0.5, markersize=2,  label=r"$f_i \cdot p_i$")
plt.plot(fp2tr_mo(odrout_mo.beta, x_.T)-f_tr,'.', alpha=0.5, markersize=2,  label=r"$f_i \cdot p_i + f_0$")
plt.plot(fp2tr_p(odrout_p.beta, x_.T)-f_tr,  '.', alpha=0.5, markersize=2,  label=r"$f_i + p_i$")
plt.plot(fp2tr(odrout_.beta, x_.T)-f_tr,     '.', alpha=0.5,  markersize=2, label=r"$f_i \cdot p_i + p_i$")
#plt.plot(fp2trf(odrout_f.beta, x_.T)-f_tr,  '.', alpha=0.5,  markersize=2, label=r"$p_i * p_i(f_i - p_i)$")
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
for probe in range(17):
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
plt.show()




'''
import keras
import tensorflow as tf

x_ = ip.fp_freq[skip:-skipb, s]
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
'''
