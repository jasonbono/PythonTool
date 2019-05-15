# Base Class for handling gm tier1 root files
# used for Trolley and FixedProbes
from __future__ import print_function
from ROOT import gInterpreter, gSystem, TChain
import os
import sys
import numpy as np
import gm2
if 'GM2' not in os.environ:
    raise Exception('The environment variable GM2 is not set.')
if 'ARTTFSDIR' not in os.environ:
    raise Exception('The environment variable ARTTFSDIR is not set.')


class rootbase:
    """ Base Class to load root files with gm2 field data.

    Attributes:
        test (bool): test
    """
    def __init__(self, dictname, prefix=None, qtname = None):
        path = os.environ['GM2']
        gInterpreter.ProcessLine('#include "'+path+'/lib/'+dictname+'.hh"')
        gSystem.Load(path+'/lib/lib'+dictname)
        if qtname is not None:
            gInterpreter.ProcessLine('#include "'+path+'/lib/'+qtname+'.hh"')
            gSystem.Load(path+'/lib/lib'+qtname)
        self.dictname = dictname
        self.qtname   = qtname

        # load common settings
        self.chain = TChain(dictname)
        if qtname is not None:
            self.qtchain = TChain(qtname)
        self.__loadSettings(prefix=prefix)
       
    def __loadSettings(self, prefix=None):
        self.basedir       = os.environ['ARTTFSDIR']+"/" #"/data1/newg2/DataProduction/Nearline/ArtTFSDir/"
        if prefix is None:
            self.fname_prefix  = "FieldGraphOut"
        else:
            self.fname_prefix  = prefix
        self.fname_suffix  = "_tier1.root"

    def loadFiles(self, runs=[]):
        for r in runs:
            self.add(runs)
        for r in self.runs:
            self.chain.Add(self.basedir+self.fname_prefix+"%05i"%r+self.fname_suffix+"/"+self.fname_path)
            if self.qtname is not None:    
                self.qtchain.Add(self.basedir+self.fname_prefix+"%05i"%r+self.fname_suffix+"/"+self.qtname_path)
        exec("from ROOT import "+self.dictname)
        exec("self.data = "+self.dictname+"(self.chain)")
        if self.qtname is not None:
            exec("from ROOT import "+self.qtname)
            exec("self.qt = "+self.qtname+"(self.qtchain)")
            self.chain.AddFriend(self.qtchain)


    def add(self, runs):
        if type(runs) == list:
            self.runs += runs
        elif type(runs) == tuple:
            self.runs += list(runs)
        else:
           self.runs.append(runs)

    def activateBranches(self, brs):
        self.chain.SetBranchStatus("*",0)
        for br in brs:
            self.chain.SetBranchStatus(br, 1)

    def checkProbe(self, prob):
        return True if ( prob >= 0 )&( prob < self.n_probes ) else False

    def getEntry(self, no):
        self.data.GetEntry(no)

    def load(self, no):
        self.getEntry(no)

    def getEntries(self):
        return self.chain.GetEntries()

    def loop(self, func, *args):
        """ loop to be modified by doughters to generate appropriate selections """
        return self.theLoop([], func, *args)

    def theLoop(self, sel, func, *args):
        """ the main loop functions, uses the selctions constrcuted in loop() """
        nentries = self.getEntries()
        self.getEntry(1)
        if len(sel) != self.n_probes:
            sel = np.full([self.n_probes], True)
        tmp_ = func(*args)
        results = []
        for r in tmp_:
            if hasattr(r, "__len__"):
                if r.shape == sel.shape:
                    results.append(np.zeros([nentries] + list(r[sel].shape)))
                else:
                    results.append(np.zeros([nentries] + list(r.shape)))
            else:
                results.append(np.zeros([nentries] + list([1])))

        for ev in range(1, nentries): # skip the first event, its usually not complete
            self.getEntry(ev)
            if ev % 10 == 0:
                print("\rReading event "+str(ev)+"/"+str(nentries)+" " + "%0.2f" % (100.0 * ev/nentries)+"%", end=' ')
                sys.stdout.flush()
            result_ = func(*args)
            for i in range(len(result_)):
                if hasattr(result_[i], "__len__"):
                    if result_[i].shape == sel.shape:
                        results[i][ev,:] = result_[i][sel]
                    else:
                        results[i][ev,:] = result_[i][:]
                else:
                    results[i][ev,:] = result_[i]
        print("\nloop done: 100%                                    ")
        return results

    # utils
    def toInt(self, i):
        if type(i) is int:
            return [i]
        elif type(i) is str:
            return [ord(i)]
        elif (type(i) is list):
            l = []
            for j in i:
                l.append(self.toInt(j)[0])
            return l
        else:
            raise ValueError('toInt() Input is not of type int or str nor a list of such.')

