# Class Based on https://arxiv.org/pdf/1805.12163.pdf
# Written by Simon Corrodi, Octobre 2018
import numpy as np
from gm2 import PPM_HZ

class Fourier(object):
    def __init__(self, phis = None, freqs = None, N=None):
        self.debug = False
        self.c     = None # coefficiencts
        self.Sinv  = None # pseudo inverse S
        if (phis is not None)|(freqs is not None)|(N is not None):
            if (phis is None)|(freqs is None)|(N is None):
                raise Exception("Either none or all of phis, freq and N need to be specified")
            else:
                self.init(phis, freqs, N)

    def init(self, phis, freqs, N):
        self.N = N
        self.updateSinv(phis)
        self.updateC(freqs)

    def f(self, n, phi):
        return np.cos(n//2*phi) * ((n+1)%2) + np.sin(n//2*phi) * (n%2)

    def updateSinv(self, phis, N=None):
        if N is not None:
            self.N = N
        if self.debug:
            print("Generate: S")
        S = self.f(np.arange(self.N * 2)[None,:], phis[:,None])
        if self.debug:
            print("done")
            print("Start S inversion")
        self.Sinv = np.linalg.pinv(S)

    def updateC(self, freq):
        if self.Sinv is None:
            raise Exception('Sinv needs to be loaded first. Call updateSinv(self, phis, N)')
        self.c = self.Sinv.dot(freq)

    def getChi(self, phis, freqs):
        return 1./phis.shape[-1] * ( ( (self.B(phis)-freqs)/PPM_HZ )**2 ).sum()

    def B(self, phi):
        if self.c is None:
            raise Exception("Fourier is not initiialized.")
        return (self.c * self.f(np.arange(self.c.shape[-1]), phi[:,None])).sum(axis=1)

    def convTest(self, phis, freqs, Nmax=1000, Nmin=5, Nstep=5, dN=[]):
        Ns       = np.arange(Nmin, Nmax, Nstep)
        chi2s    = np.empty_like(Ns, dtype='double') 
        chi2bars = np.full([Ns.shape[-1], len(dN)], np.nan) 
        cs = []

        def chi2bar(i, j):
            return 0.5 * (2.0 * (cs[i][0]-cs[j][0])**2 +  ((cs[i][2:Ns[j]*2] - cs[j][2:Ns[j]*2])**2).sum())/PPM_HZ**2

        for i, N in enumerate(Ns):
            self.init(phis, freqs, N)
            chi2s[i] = self.getChi(phis, freqs)
            if len(dN) > 0:
                cs.append(self.c)
                for j, DN in enumerate(dN):
                    if i >= DN//Nstep:
                        chi2bars[i, j] = chi2bar(i, i - DN//Nstep)
        return Ns, chi2s, chi2bars

