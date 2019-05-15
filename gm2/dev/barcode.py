import numpy as np

import Trolley

rn = [5217]
cw = False

def decode(data, downstream=True):
    if len(data) != 13:
        print "Barcode Data has a length of 12 bits"
        return -1
    d = data < (data.min() + data.max())/2.0
    if (not downstream):
        d = d[np.concatenate((np.arange(11,-1,-1),[12]))]
    if (not d[0])|(not d[11])|(d[12]):
        #plt.plot(data,'-o')
        #plt.show()
        print "bc error?",
        #return -1
    b8_   = np.packbits(d[1:11])
    return int(b8_[0]) * 4 + b8_[1]/64
    #if lend:
    #    return b8_[0] + 2**8 + int(b8_[1])
    #else:
    #    return b8_[1] + 2**8 + int(b8_[0])

def getBarcode(start, nframes):
    tr = np.zeros([6,0])
    ts = np.zeros([0])
    for ev in np.arange(start, start+nframes):
        tr = np.concatenate((tr, trs[ev]), axis=1)
        ts = np.concatenate((ts, trs_ts[ev]), axis=0)
        s = (tr[0,:]>1.0)&(tr[0,:]<5.0)
    return tr[:,s], ts[s]


t = Trolley.Trolley(rn)

def callback():
    return  t.getBarcodeTrace(), t.getBarcodeTs()
trs, trs_ts = t.loop(callback)



s = (trs[:,0,:].reshape([-1])>1.0)&(trs[:,0,:].reshape([-1])<5.0)

s2 = s#(trs[:,3,:].reshape([-1])>1.0)&(trs[:,3,:].reshape([-1])<5.0)
clk = [trs[:,1,:].reshape([-1])[s], trs[:,4,:].reshape([-1])[s]]
ab  = [trs[:,0,:].reshape([-1])[s], trs[:,3,:].reshape([-1])[s]]
t   = trs_ts.reshape([-1])[s]


from scipy import signal
# no sucess with fourier
'''
n = 25000
#f, t, Zxx = signal.stft(clk-clk.mean(), nperseg=n, noverlap=0)
#freq_index = np.argmax(np.abs(Zxx)[2:], axis=0)+2
#freq  = f[freq_index]
#phase = np.angle(freq_index)

for i in range(int(clk.shape[-1]/n)):
    if i < 10:
        continue
    if i > 20:
        break
    start = i     * n
    end   = (i+1) * n
    wf = clk[start:end]-clk[start:end].mean()
    plt.plot(np.arange(start, end), wf , c='blue')
    sp = np.fft.fft(wf)
    freq_index = np.argmax(np.abs(sp))
    freq  = np.fft.fftfreq(n)[freq_index]
    phase = np.angle(sp)[freq_index]
    plt.plot(np.arange(start, end), np.cos(2*np.pi * freq * np.arange(n) + phase), c='red')

'''
offset = 100000
freq_cut = 0.001
b, a = signal.butter(3, freq_cut)
y = [signal.filtfilt(b, a, clk[0][offset:]), 
     signal.filtfilt(b, a, clk[1][offset:])]
zeros = [np.where(np.diff(np.sign(np.diff(y[0], n=1))))[0], 
         np.where(np.diff(np.sign(np.diff(y[1], n=1))))[0]]

import matplotlib.pyplot as plt
'''
plt.plot(clk[offset:])
plt.plot(zeros, clk[zeros+offset], 'o')
plt.plot(ab[offset:])
plt.plot(zeros, ab[zeros+offset], 'o')
'''

#zeros_sep = 600
th_g = 0.6
th_l = 0.3
th_clk = 0.03
cnt = 0
codes       = [[],[]]
code_word   = [[],[]]
code_num    = [[],[]]
errors      = [[],[]]
removed     = [[],[]]
for j in range(2):
    code_i = 0
    zero_last = zeros[j][0]
    for i, index in enumerate(zeros[j]):
        #if i  > 0:
            #print np.abs(clk[index+offset]-clk[zeros[i-1]+offset]), np.abs(clk[index+offset]-clk[zeros[i-1]+offset]) < th_l
            #if np.abs(clk[index+offset]-clk[zeros[i-1]+offset]) < th_clk:
            #    removed.append(index)
            #    continue
        if i > 5:
            h = ab[j][zeros[j][i-5:i]+offset]
            m = h.mean()
            s = h.std()
            #if (np.abs(ab[index]-m) > th*s)&((i-code_i)>=10):
            if ((m-ab[j][index+offset]) > th_g)&((ab[j][zeros[j][i-1]+offset]-ab[j][index+offset]) > th_l)&((i-code_i) >= 15):
                # new code
                if (cnt != 30)&(len(codes[j])>0):
                    errors[j].append(codes[j][-1])
                    print "From", zeros[j][ codes[j][-1]], "to", index, cnt, "peaks are obsearved."
                cnt = 0
                code_i = i
                codes[j].append(i)
                code_word[j].append(ab[j][zeros[j][i:i+13] + offset])
                code_num[j].append(decode(code_word[j][-1], cw))
                print j, len(codes[j])-1, ") at", zeros[j][i], ":", decode(code_word[j][-1], cw)
            cnt = cnt +1
    print "Error rate:", 1.0*len(errors[j])/len(codes[j])


'''
no = 0
plt.plot(clk[no][offset:], alpha=0.5)
plt.plot(zeros[no], clk[no][zeros[no]+offset], '.', alpha=0.5)
removed = np.array(removed)
if removed[no].shape[0] > 0:
    plt.plot(removed[no], clk[no][removed[n]+offset], '.', alpha=0.5)
plt.plot(ab[no][offset:])
plt.plot(zeros[no], ab[no][zeros[no]+offset], '.', alpha=0.5)
codes = np.array(codes)
plt.plot(zeros[no][codes[no]], ab[no][zeros[no][codes[no]]+offset], 'o')
errors = np.array(errors)
plt.plot(zeros[no][errors[no]], ab[no][zeros[no][errors[no]]+offset], 'd', color='red')
plt.show()
'''
map_t = [np.full([30 * 2**10], -1), np.full([30 * 2**10], -1)]

code_status = [[],[]]
code_time   = [[],[]]
sign = 1 if cw else -1
for j in range(2):
    for i in range(len(code_num[j])):
        status = 0
        if i > 0:
            if codes[j][i] in errors[j]:
                status = status + 1*sign # not 30 counts to next abs
            if code_num[j][i-1] + 1*sign != code_num[j][i]:
                # try to correct difference to last abs number
                if (i<len(code_num[j])-2):
                    #if (code_status[j][i-2] % 2 == 0)&(code_status[j][i-1] % 2 == 0):
                    #    print code_status[j][i-2], code_status[j][i-1], code_num[j][i-2]+2, code_num[j][i-1]+1, code_num[j][i], code_num[j][i+1]-1, code_num[j][i+2]-2
                    if ((code_status[j][i-2] == 0)|\
                        (code_status[j][i-2] == 2))&\
                        (code_status[j][i-1] == 4)&\
                        (code_num[j][i-2] + 1*sign == code_num[j][i-1])&\
                       ((code_num[j][i+1] - 2*sign == code_num[j][i-1])|\
                        (code_num[j][i+2] - 3*sign == code_num[j][i-1])):
                           code_num[j][i] = code_num[j][i-1]+1*sign
                           #print "fix a code number"
                status = status + 2 # difference to last abs number
            if i < len(code_num[j])-1:
                if code_num[j][i] + 1*sign != code_num[j][i+1]:
                    status = status + 4 # difference to next abs number

        code_status[j].append(status)
        code_time[j].append(t[zeros[j][codes[j][i]]+offset])
        
        # plot
        if status % 2 == 0: 
            if status/2 != 3:
                if i > 1:
                    if code_status[j][-2]/2 == 3:
                        for k in range(30):
                            #map_s[j].append(code_num[j][i-1] + k/30.0)
                            map_t[j][(code_num[j][i]-1)*30 + k*sign] = t[zeros[j][codes[j][i-1] + k] + offset]
                        plt.plot([t[zeros[j][codes[j][i-1]]+offset], code_time[j][-1]], [j, j], '-', color='red')
                if i < len(code_num[j])-1:
                    for k in range(30):
                        #map_s[j].append(code_num[j][i] + k/30.0)
                        map_t[j][code_num[j][i]*30 + k*sign] = t[zeros[j][codes[j][i] + k] + offset]
                    plt.plot([code_time[j][-1], t[zeros[j][codes[j][i+1]]+offset]], [j, j], '-', color='blue')
plt.show()

#code_status

plt.plot(map_t[0])
plt.plot(map_t[1])
plt.show()

#plt.show()



# user a filter

