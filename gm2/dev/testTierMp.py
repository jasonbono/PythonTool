import gm2



def testMultipole(tr, event): 
    tr.load(event-1) 
    azi99 = gm2.np.array(tr.getPhi(2)) 
    f99   = gm2.np.array(gm2.np.frombuffer(tr.data.Quality_FrequencySmoothed)) 
    tr.load(event) 
    azi100 = gm2.np.array(tr.getPhi(2)) 
    f100   = gm2.np.array(gm2.np.frombuffer(tr.data.Quality_FrequencySmoothed)) 
    tr.load(event+1) 
    azi101 = gm2.np.array(tr.getPhi(2)) 
    f101   = gm2.np.array(gm2.np.frombuffer(tr.data.Quality_FrequencySmoothed)) 

    f = gm2.np.zeros([17]) 
    phi = azi100[8]
    f[8] = f100[8]
    for p in range(17): 
        if p > 8:     
            dphi = azi100[p] - azi99[p]  
            dfreq = f100[p] - f99[p]  
            print((phi-azi99[p])/dphi)  
            f[p] = f99[p] + dfreq * (phi-azi99[p])/dphi  
        if p < 8: 
            dphi = azi101[p] - azi100[p] 
            dfreq = f101[p] - f100[p] 
            print((phi-azi100[p])/dphi) 
            f[p] = f100[p] + dfreq * (phi-azi100[p])/dphi 
    fit = gm2.util.multipoleJoe(f)
    mp = gm2.np.array([fit.GetParameters()[i] for i in range(1,10)])
    tr.load(event)
    return mp-tr.getMultipole()
