import numpy as np
import gm2

from datetime import date, datetime, timedelta
from time import mktime

def date2ts(year, month, day):
    return mktime(date(year, month, day).timetuple())

def datetime2ts(year, month, day, h = 0, m = 0, s = 0):
    return datetime2ts_dt(datetime(year, month, day, h, m, s))

def datetime2ts_dt(dt):
    return  mktime(dt.timetuple()) * 1e9

def ts2datetime(ts):
    try:
        if len(ts) == 0:
             return ts
    except:
        pass
    return np.vectorize(datetime.fromtimestamp)(ts/1e9)

def nearestIndex(arr, v):
    return (np.abs(arr - v)).argmin()

def firstIndexBelowThr(wf, thr, start):
    if start >= len(wf):
        return 0
    index = np.argmax(wf[start:]<thr) + start
    if (index == 0)|(index == start):
        return np.nan
    #print(thr, index, wf[index], wf[index+1])
    return index if wf[index + 1] < thr else firstIndexBelowThr(wf, thr, index + 1)

def firstIndexAboveThr(wf, thr, start=0):
    return firstIndexBelowThr(-wf, -thr, start)

def edge(wf_t, wf_v, th):
    index = firstIndexAboveThr(wf_v, th)
    if np.isnan(index):
        return np.nan
    if index == 0:
        return np.nan
    fraction = (th - wf_v[index-1])/(wf_v[index] - wf_v[index-1])
    return wf_t[index-1] + fraction * (wf_t[index] - wf_t[index-1])

def cf(wf_t, wf_v, fraction=0.5, returnTh=False):
    th = wf_v.min() + (wf_v.max() - wf_v.min()) * fraction
    if returnTh:
        return edge(wf_t, wf_v, th), th
    else:
        return edge(wf_t, wf_v, th)

def cf_local(wf_t, wf_v, fraction=0.5, returnTh=False, baseline_w=16, baseline_o=30):
    '''cf of wf where the minimum of the wf is determined by the mean at -baseline_o +/- baseline_w/2 and the single max is used.
       developed to find the trolley in fp waveforms'''
    th = wf_v.mean() + (wf_v.max() - wf_v.mean()) * fraction
    index = firstIndexAboveThr(wf_v, th)
    if np.isnan(index):
        if returnTh:
            return np.nan, np.nan
        else:
            return np.nan
    index_min = int(index - baseline_o - baseline_w/2)
    if index_min < 0:
        index_min = 0
    index_max = int(index - baseline_o + baseline_w/2)
    if index_max < 0:
        index_max = 0
    if index_max > wf_v.shape[-1]:
        index_max = wf_v.shape[-1]
    baseline = wf_v[index_min:index_max].mean()

    th = baseline + (wf_v.max() - baseline) * fraction
    index_max = int(index + baseline_o + baseline_w/2)
    if index_max > wf_v.shape[-1]:
        index_max = wf_v.shape[-1]
    if returnTh:
        return edge(wf_t[index_min:index_max], wf_v[index_min:index_max], th), th
    else:
        return edge(wf_t[index_min:index_max], wf_v[index_min:index_max], th)


from scipy import signal
def smooth(wf, l, mode='box'):
    if mode == 'box':
        g = np.ones([l])
        #return np.convolve(wf, 1.0*np.ones([l])/l, mode='valid')
    if mode == 'gauss':
        g = signal.gaussian(11, std=l)
       # return np.convolve(wf, 1.0*g/g.sum(), mode='valid')
    if mode == 'lr':
        d = l/2+1
        g = (1.0-np.abs(np.arange(-d, d+1,1.0)/(1.0*d))**3)**3
    if mode == 'ref':
        g = np.zeros(l)
        g[l/2] = 1
    return np.convolve(wf, 1.0*g/g.sum(), mode='valid')

def gauss(x, N, mean, sigma):
    return N / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mean)/sigma)**2 )
        

def multipole(coord, *p):
    r     = coord[0]
    theta = coord[1]
    #B = p[0]
    #print(len(p), p)
    if len(p) == 1:
     return p[0]
    #print(B, p)
    #print p, B, len(p)
    #if len(c) != len(s):
    #    raise ValueError("c("+str(len(c))+") and s("+str(len(s))+") require the same length")
    n = np.arange(1, len(p)//2+1)
    #print("DEBUG", r.shape, theta.shape, p[0], p[1::2], p[2::2], n)
    return p[0] + ( (r[:,None]**(n)) * (np.array(p[1::2]) * np.cos(n * theta[:,None]) + np.array(p[2::2]) * np.sin(n * theta[:,None]))).sum(axis=1)
    #for n in range(1, len(p)//2+1):
    #    #print n, p[2*n-1], p[2*n]
    #    B = B + r**(n) * (p[2*n-1] * np.cos(n * theta) + p[2*n] * np.sin(n * theta))
    #return B


from scipy.optimize import curve_fit
def getTrMultipole(probes, n=9, at=45.0):
    """Fit multipole expension to trolley probe frequency data.
    
    This function uses the default trolley probe positions stored in gm2.TR.probes.

    Args:
        probes (numpy.array(float, [17])) : trolley probe frequencies.
        n (int, optional) : mutipole order. Defaulkts to 9.
        at (float, optional) : radius of multipole evaluation in mm. Defaults to 45mm.

    Returns:
        numpy.array (float, [n]) : multipole coefficents evaluated at radius at.

    Examples:

        >>> import gm2
        >>> tr = gm2.Trolley([3997])
        >>> tr.load(100) # load desired root entry/trolley cycle
        >>> freq = tr.getFrequency(0)
        >>> multipoles = gm2.util.getTrMultipole(freq)
        >>> for i, mp in enumerate(multipoles):
        >>>     print "%i) %.3f" % ( i, mp)

    """
    p0 = np.concatenate(([probes.mean()], np.zeros([n-1])))
    popt, _ = curve_fit(multipole, xdata=(gm2.TR.probes.position.r, gm2.TR.probes.position.theta), ydata=probes, p0=p0)
    return popt * np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5, at**6, at**6, at**7, at**7, at**8, at**8])[:n] 
    #return multipole((MP.r, 0), popt)

def getFpMultipole(pos, freq, at=45.0, sigma=None, n=-1):
    if n <= 0:
        n_ = freq.shape[0]
    else:
        n_ = n
    p0 = np.concatenate(([freq.mean()], np.ones([n_ - 1])))
    rr = np.array([at**0, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5])
    if sigma is None:
        popt, _ = curve_fit(multipole, xdata=pos, ydata=freq, p0=p0)
        return popt * rr[:popt.shape[0]]
    else:
        popt, pcov = curve_fit(multipole, xdata=pos, ydata=freq, p0=p0, sigma=sigma, absolute_sigma=True)
        return popt * rr[:popt.shape[0]], np.sqrt(np.diag(pcov)) * rr[:popt.shape[0]]

from scipy.optimize import curve_fit 
def func_lin(x, a, b):
    return a + b * x

def f_lin(B, x):
    return B[0] + B[1] * x

def fit_lin(x, y, yerr=None, p0=None):
        if yerr:
            popt, pcov = curve_fit(func_lin, x, y, p0=p0)
        else:
            popt, pcov = curve_fit(func_lin, x, y, sigma=yerr, p0=p0)
        return popt

def getBinCenter(bins):
    return bins[:-1] + np.diff(bins) / 2.0

def fit_gauss(v, bc, p0=None):
    s  = v>0
    if p0 is None:
        p0 = [v.sum(), 0, 10.0]
    return curve_fit(gauss, bc[s], v[s], p0=p0, sigma=np.sqrt(v[s]), absolute_sigma=True)

from scipy.interpolate import interp1d
from scipy import odr
class FitLin:
    def __init__(self, x, y, xerr=None, yerr=None):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr

    def f(self, B, x):
        return B[0] + B[1] * x

    def fit(self, beta0=None, ifixb=None):
        if self.xerr:
            sx=self.xerr
        else:
            sx = 1.0
        if self.yerr:
            sy = self.yerr
        else:
            sy = 1.0
        self.odr = odr.ODR(odr.Data(self.x, self.y, wd=1./sx**2, we=1./sy**2), odr.Model(self.f), beta0=beta0, ifixb=ifixb)
        self.odrout = self.odr.run()

    def getPara(self):
        return self.odrout.beta, self.odrout.sd_beta

    def getCov(self):
        return self.odrout.cov_beta

    def getY(self, xx=None):
        if xx is None:
            xx = self.getXX();
        return self.f(self.odrout.beta, xx)

    def getXX(self, xadd=0, n=200):
        dx = (self.x.max() - self.x.min())/200.
        return np.arange(self.x.min()-xadd, self.x.max()+xadd+dx, dx)

    def getFit(self, xadd=0, n=200):
        xx = self.getXX(xadd, n)
        return xx, self.getY(xx)

    def getBand(self, xx=None, xadd=0):
        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
        C = self.getCov()
        if xx is None :
            xx = self.getXX()
        self.xx = xx
        df2 = C[0, 0] + self.xx * (C[1, 0] + C[0, 1]) + self.xx**2 * C[1,1]
        from scipy.stats import t
        dof = self.x.shape[0]-2
        tval = t.ppf(1.0-(1.0 - 0.683)/2., dof)
        covscale = 1.0
        covscale = self.odrout.sum_square/dof
        return tval * np.sqrt(covscale * df2)

def multipoleJoe(freqs, nPars = 10):
    """Calculate Multipoles in joe-method.
    
    This is the same code/method as used in the art nearline production.

    Args:
        freq[17] (np.array(float)): 17 probe frequencies.
        nPars (int): number of multipole coefficients.

    Returns:
        f (ROOT.TF2): fitted multipole function.
    
    Example:

        Fit one trolley cylce:

        >>> import gm2
        >>> tr = gm2.Trolley([3997])
        >>> tr.load(100) # load the desired root event/trolley cycle
        >>> freq = tr.getFrequency(0)
        >>> tf2 = gm2.util.multipoleJoe(freq)
        >>> for n in range(tf2.GetNpar()):
        >>>    print(tf2.GetParName(n), tf2.GetParameter(n))

    """

    class myRFunc:
        """Wrapper for ROOT.TF2 to be used in joe-style multipole fits."""
        def __init__(self, tag, lower_range_x=-4.5, upper_range_x=4.5, lower_range_y=-4.5, upper_range_y=4.5, nPara=10):
            self.function = gm2.ROOT.TF2(tag, self, lower_range_x, upper_range_x, lower_range_y, upper_range_y, nPara)
            self.function.SetParNames("NPars","Dipole","Normal quad","Skew quad","Normal sext","Skew sext","Normal oct","Skew oct","Normal deca","Skew deca")
            self.function.FixParameter(0,nPars)
        def __call__(self, x, par):
            x_ = x[0]
            y_ = x[1]
            nP = par[0]
            B = gm2.np.copy(par[1])
            r = gm2.ROOT.TMath.Sqrt(x_**2 + y_**2)
            rf = r/4.5
            theta = gm2.ROOT.TMath.ATan2(y_,x_)
            for i in range(2, int(nP), 2):
                B += gm2.ROOT.TMath.Power(rf,1*(i//2)) * (par[i] * gm2.ROOT.TMath.Cos((i//2)*theta) + par[i+1] * gm2.ROOT.TMath.Sin((i//2)*theta))
            if (r<4.5)|(r==4.5):
                return B
            else:
                return 0.0
        def TF2(self):
            return self.function

    f = myRFunc('myRFunc',-4.5,4.5,-4.5,4.5, nPars)
    f.TF2().FixParameter(0,nPars)
    #f.SetParNames("NPars","Dipole","Normal quad","Skew quad","Normal sext","Skew sext","Normal oct","Skew oct","Normal deca","Skew deca");
    #f.FixParameter(0,nPars)

    gTCalib = gm2.ROOT.TGraph2D()
    x = gm2.TR.probes.joePosition.ProbePosX
    y = gm2.TR.probes.joePosition.ProbePosY
    for j in range(17):
        gTCalib.SetPoint(j,x[j],y[j],freqs[j])
    gTCalib.Fit(f.TF2(),"Q");
    return f.TF2()

import os
def fileExists(run):
    """Checks if the standard root files of the given run(s) exists. 
    Args:
        run(int) : run number.
    
    Retuerns:
        exists(bool): True if the file exists.
    """
    basedir       = os.environ['ARTTFSDIR']+"/" #"/data1/newg2/DataProduction/Nearline/ArtTFSDir/"
    fname_prefix  = "FieldGraphOut"
    fname_suffix  = "_tier1.root"
    fname = basedir+fname_prefix + ("%05i" % run) + fname_suffix
    return os.path.exists(fname)


