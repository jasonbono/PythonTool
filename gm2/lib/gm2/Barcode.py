import gm2


class Barcode(object):
    def __init__(self, runs=[], cw=False):
        print("WARNING. This is very experimental :-)")
        self.cw = cw
        self.tr = gm2.Trolley(runs)
        if len(runs) > 0:
            self.load();
        self.hasNums = False
        self.verbosity = 1

    class Code:
        def __init__(self, index, clk_n):
            self.index = index  # bc wf index
            self.clk_n = clk_n  # clk number
            self.status = 0
            self.clk_last = 0

        def addCode(self, data, cw=False):
            self.code_a   = data             # analogue code in binary format
            self.code_num = self.decode(cw)  # decoded code as dec number
            self.code_no  = -1               # code number (position in ring)
            self.code_no_status = 0          # status of code2num matching 

        def decode(self, downstream=False):
            if len(self.code_a) != 13:
                print("Barcode Data has a length of 12 bits")
                return -1
            d = self.code_a < (self.code_a.min() + self.code_a.max())/2.0
            if (not downstream):
                d = d[gm2.np.concatenate((gm2.np.arange(11,-1,-1),[12]))]
            #if (not d[0])|(not d[11])|(d[12]):
            #    #gm2.plt.plot(data,'-o')
            #    #gm2.plt.show()
            #    print("bc error?")
            #    #return -1
            b8_   = gm2.np.packbits(d[1:11])
            return int(b8_[0]) * 4 + b8_[1]/64

    def plotCode(self, j, i, tminus=10.0, tplus=6.0):
        code = self.codes[j][i]
        t0 = self.time[code.index]
        s_plot =  (self.time           > t0 - tminus * 1e9) & (self.time           < t0 + tplus * 1e9)
        sz     =  (self.time[self.clk_index[j]] > t0 - tminus * 1e9) & (self.time[self.clk_index[j]] < t0 + tplus * 1e9)

        gm2.plt.plot(self.time[s_plot]-t0,       self.clk[j][s_plot], alpha=0.3, color=gm2.colors[0])
        gm2.plt.plot(self.time[self.clk_index[j]][sz]-t0, self.clk[j][self.clk_index[j]][sz], '.', alpha=0.3, color=gm2.colors[0])
        gm2.plt.plot(self.time[s_plot]-t0,       self.abs[j][s_plot], color=gm2.colors[1])
        gm2.plt.plot(self.time[self.clk_index[j]][sz]-t0, self.abs[j][self.clk_index[j]][sz], '.', color=gm2.colors[2])
        codes_index = [code.index for code in self.codes[j]]
        sc     = ((self.time[codes_index] > t0 - tminus * 1e9) & 
                  (self.time[codes_index] < t0 + tplus * 1e9))
        gm2.plt.plot(self.time[codes_index][sc]-t0, self.abs[j][codes_index][sc], '^', color=gm2.colors[3])
        gm2.despine()
        gm2.plt.show()

    def load(self, freq_type = 0, plot=False):
        self.tr.loadBarcode(add=["ProbeFrequency","TimeStamp","Position"])
        def tr_callback():
            return [self.tr.getTimeGPS(), self.tr.getPhi(0), self.tr.getFrequency(freq_type), self.tr.getTimeBarcodes()]
        
        #self.tr.activateBranches(["ProbeFrequency","TimeStamp","Position"])
        self.tr_time, self.tr_pos, self.tr_freq, bc_t = self.tr.loop(tr_callback)
        self.clk = [self.tr.bc[2,:], self.tr.bc[5,:], self.tr.bc[1,:], self.tr.bc[4,:]]
        self.abs = [self.tr.bc[3,:], self.tr.bc[6,:]]
        self.time = self.tr.bc[0,:]
        #bc_s = (bc_trcs[:,0,:].reshape([-1])>1.0)&(bc_trcs[:,0,:].reshape([-1])<5.0)
        #self.clk = [bc_trcs[:,1,:].reshape([-1])[bc_s],
        #            bc_trcs[:,4,:].reshape([-1])[bc_s]]
        #self.abs = [bc_trcs[:,0,:].reshape([-1])[bc_s],
        #            bc_trcs[:,3,:].reshape([-1])[bc_s]]

        if(True):
            probe_ = 0
            s = (bc_t[1:-1,probe_] > 1e5)&(self.tr_time[1:-1, probe_] > 1e5)
            popt_ = gm2.util.fit_lin(bc_t[1:-1,probe_][s], self.tr_time[1:-1, probe_][s], p0=[self.tr_time[1:-1, probe_][s].min(), 1e6])
            if plot:
                gm2.plt.subplot(211)
                gm2.plt.plot(self.tr_time[1:-1, probe_][s], bc_t[1:-1,probe_][s],'.')
                gm2.plt.plot(gm2.util.func_lin(bc_t[1:-1,probe_],popt_[0],popt_[1]), bc_t[1:-1,probe_])
                gm2.plt.xlabel("gps times [ns]")
                gm2.plt.ylabel("barcode times [ns]")
                gm2.plt.subplot(212)
                gm2.plt.plot(bc_t[1:-1,probe_][s], gm2.util.func_lin(bc_t[1:-1,probe_][s],popt_[0],popt_[1]) - self.tr_time[1:-1, probe_][s])
                gm2.plt.xlabel("gps time [ns]")
                gm2.plt.ylabel("residuals [ns]")
                gm2.despine()
                gm2.plt.show()
                print(popt_)
            if(abs(popt_[1] - 1.0) > 0.001):
                raise ValueError('barcode timestamp conversion failed')
        else:
            popt_ = [0, 1e6]
        self.time = self.time * popt_[1] + popt_[0]


    def loadClkIndex(self, plot=False):
        if self.verbosity > 0:
            print("load clocks")
        #print("Filter Barcodes")
        freq_cut = 0.001
        from scipy import signal
        b, a = signal.butter(3, freq_cut)
        smooth_clk = [signal.filtfilt(b, a, self.clk[0]),
                         signal.filtfilt(b, a, self.clk[1])]
        #bc_smooth_abs = [signal.filtfilt(b, a, bc.bc_abs[0]),
        #                 signal.filtfilt(b, a, bc.bc_abs[1])]
        self.clk_index = [gm2.np.where(gm2.np.diff(gm2.np.sign(gm2.np.diff(smooth_clk[0], n=1))))[0],
                      gm2.np.where(gm2.np.diff(gm2.np.sign(gm2.np.diff(smooth_clk[1], n=1))))[0]]

        cut = 0.1
        for j in range(2):
            self.clk_index[j] = gm2.np.delete(self.clk_index[j], gm2.np.argwhere(gm2.np.abs(gm2.np.diff(self.clk[j][self.clk_index[j]])) < cut))


        if plot:
            s  = (self.time > self.time[6000000]) & (self.time < self.time[6100000])
            sz = (self.time[self.clk_index[0]] > self.time[6000000]) & (self.time[self.clk_index[0]] < self.time[6100000])
            gm2.plt.plot(self.time[s], self.clk[0][s])
            gm2.plt.plot(self.time[self.clk_index[0]][sz], self.clk[0][self.clk_index[0]][sz], '.')
            gm2.plt.xlabel("time [ns]")
            gm2.plt.ylabel("barcode [V]")
            gm2.despine()
            gm2.plt.show()


    def loadCodes(self, nplot=0):
        if self.verbosity > 0:
            print("load codes")
        # requires clk_index
        if not hasattr(self, 'clk_index'):
            self.loadClkIndex()
        th_g = 0.6
        th_l = 0.3
        th_clk = 0.03
        codes       = [[],[]]  # stores clk number of abs group (code) start
        #code_word   = [[],[]]  # stores         code group (13 bits, 12 are code, one is used to check that it is off agian)
        #code_num    = [[],[]] # stores number from decoded code group
        errors      = [0, 0]  # 
        #removed     = [[],[]]
        self.codes = [[],[]]

        window = 5
        skip   = window # >= window 
        test = 0

        for j in range(2):
            forceCode = False
            clk_n_last_code  = 0                          # clock number of last abs groupe (code)
            clk_since_code   = 0                          # counts clk since last abs groupe
            #z_index_last    = zeros[j][0]                # index of 
            code_cnt        = 0                           # counts abs group  (code_i)
            for clk_n, clk_index in enumerate(self.clk_index[j]):
                if clk_n <= skip:                         # skip first 5 zeros
                    continue
                #if test > 30:
                #    brek
                # calculate local mean and std to determine if a new abs group starts, always "zeros"  in between
                h_    = self.abs[j][self.clk_index[j][clk_n-window:clk_n]]
                mean_ = h_.mean()
                std_  = h_.std()

                # check for new abs group
                #if (((self.abs[j][clk_index] - mean_) < th_g)&
                #    ((self.abs[j][clk_index+11] - mean_) < th_g)):

                if ( (forceCode == True) | 
                     (((self.abs[j][clk_index] - mean_) < th_g)&                        # falling edge with respect to last [window] clks
                      ((self.abs[j][self.clk_index[j][clk_n-1]]-self.abs[j][clk_index]) > th_l)&   # relative edge
                      ((clk_n - clk_n_last_code) >= 15))):                             # at least 15 clk since last abs groupe
                    #print(self.abs[j][clk_index + 11 ] - mean_ > th_g)               
                    # check if it is acually a code and not only a spike/gap
                    if ((self.abs[j][clk_index + 11 ] - mean_ > th_g )& # if everything is allright, this is the end of the code which is always high
                        (self.abs[j][clk_index + 11 ] - mean_ > th_g )& # to alow for semi-broken codes allso check the two positions next to it
                        (self.abs[j][clk_index + 11 ] - mean_  > th_g)&
                        (forceCode == False)):
                        clk_since_code += 1
                        continue;
                    if ((forceCode == False)&
                        (clk_since_code == 29) & 
                        (self.abs[j][clk_index + 12 ] - mean_  > th_g )):
                        clk_since_code += 1
                        forceCode = True 
                        continue;
                    forceCode = False
                    # new code (abs group)
                    self.codes[j].append(self.Code(clk_index, clk_n))
                    self.codes[j][len(self.codes[j])-1].addCode(self.abs[j][self.clk_index[j][clk_n:clk_n+13]], self.cw)
                    self.codes[j][len(self.codes[j])-1].clk_last = clk_since_code
                    codes[j].append(clk_n)                          # store clk number of new code
                    #code_word[j].append(bc.abs[j][zeros[j][clk_n:clk_n+13]])    # store new code
                    if (clk_since_code != 30)&(len(codes[j])>0):                     # if not excact 30 clk since last abs group (code) -> error
                        self.codes[j][len(self.codes[j])-1].status = 1
                        if (clk_since_code + self.codes[j][len(self.codes[j])-2].clk_last == 60):
                            self.codes[j][len(self.codes[j])-1].status = 2
                            self.codes[j][len(self.codes[j])-2].status = 2
                            errors[j] -= 2
                        errors[j] += 1
                        #print("From", self.clk_index[j][ codes[j][-1]], "to", clk_index, clk_since_code, "peaks are obsearved.")
                        if (test < nplot):
                            self.plotCode(j,len(self.codes[j])-1)
                            '''t0 = self.time[clk_index]
                            tminus = 10.0 # in s
                            tplus  =  1.0 # in s
                            s_plot =  (self.time           > t0 - tminus * 1e9) & (self.time           < t0 + tplus * 1e9)
                            sz     =  (self.time[self.clk_index[j]] > t0 - tminus * 1e9) & (self.time[self.clk_index[j]] < t0 + tplus * 1e9)
                            sc     = ((self.time[self.clk_index[j][codes[j]]] > t0 - tminus * 1e9) & 
                                      (self.time[self.clk_index[j][codes[j]]] < t0 + tplus * 1e9))
                            gm2.plt.plot(self.time[s_plot]-t0,       self.clk[j][s_plot], alpha=0.3, color=gm2.colors[0])
                            gm2.plt.plot(self.time[self.clk_index[j]][sz]-t0, self.clk[j][self.clk_index[j]][sz], '.', alpha=0.3, color=gm2.colors[0])
                            gm2.plt.plot(self.time[s_plot]-t0,       self.abs[j][s_plot], color=gm2.colors[1])
                            gm2.plt.plot(self.time[self.clk_index[j]][sz]-t0, self.abs[j][self.clk_index[j]][sz], '.', color=gm2.colors[2])
                            gm2.plt.plot(self.time[self.clk_index[j][codes[j]]][sc]-t0, self.abs[j][self.clk_index[j][codes[j]]][sc], '^', color=gm2.colors[3])

                            gm2.plt.show()'''
                            test += 1

                    clk_since_code  = 0               # reset 
                    clk_n_last_code = clk_n           # letch
                    #code_num[j].append(decode(code_word[j][-1], cw))
                    #print j, len(codes[j])-1, ") at", zeros[j][i], ":", decode(code_word[j][-1], cw)
                clk_since_code += 1
            print("Error rate:", 1.0*errors[j]/len(codes[j]))

    def loadDB(self, fname="BarcodeDmod.json", correct=True):
        import os
        path = os.environ['GM2']
        import json
        with open(path+"/data/"+fname, "r") as f:
            db = json.load(f)

        #dbmap = gm2.np.array([a['AltCode'] for a in db['AbsDataArray']])
        self.dbmap = gm2.np.array([a['Code'] for a in db['AbsDataArray']])
        if correct:
            #self.dbmap[357] = 13
            #self.dbmap[356] = 718
            #self.dbmap[358] = 717
            #self.dbmap[359] = 716
            #self.dbmap[361] = 167
            #self.dbmap[363] = 161
            #self.dbmap[365] = 622
            #self.dbmap[366] = 621
            #self.dbmap[367] = 748

            pass
            '''
            self.dbmap[593] = 167
            self.dbmap[592] = 165
            self.dbmap[591] = 163
            self.dbmap[590] = 161
            self.dbmap[589] = 623
            self.dbmap[588] = 622
            self.dbmap[587] = 621
            self.dbmap[586] = 748
            self.dbmap[585] = 151

            self.dbmap[562] = 628
            self.dbmap[561] = 627
            self.dbmap[560] = 625
            self.dbmap[559] = 624
            self.dbmap[558] = 623'''


    def Code2Num(self, fname="BarcodeDB.json"):
        # requires codes (and dbmap_
        if not hasattr(self, 'codes'):
            self.loadCodes()
        if not hasattr(self, 'dbmap'):
            self.loadDB(fname)
        #self.num = [[],[]]
        #self.status = [[],[]]
        sign = 1 if self.cw else -1
        for j in range(2):
         for i  in range(len(self.codes[j])):
          nn = self.codes[j][i].code_num
          if i > 0:
            #Vprint(num[i-1], gm2.np.argwhere(dbmap==num[i-1]), jj[i-1], status[i-1])
            m = gm2.np.argwhere(self.dbmap==nn)
            if m.shape[0] == 0:
                mm = gm2.np.nan
            elif m.shape[0] == 1:
                mm = m[0][0]
            else:
                mm = m[gm2.np.abs(m - self.codes[j][i-1].code_no).argmin()][0] # clostes to last

            if mm == self.codes[j][i-1].code_no + sign:          ## perfect otder
                self.codes[j][i].code_no = mm
                self.codes[j][i].code_no_status = 1
                continue
            if abs(mm - self.codes[j][i-1].code_no - sign) < 2:  ## of by one
                self.codes[j][i].code_no = mm
                self.codes[j][i].code_no_status = 2
                j_ = 1
                while(self.codes[j][i-j_].code_no_status == 0):
                    #print(jj[i-j])
                    self.codes[j][i-j_].code_no = gm2.np.nan
                    j_ += 1
                continue
            #if mm in [654]:
            #    #print(jj[i-1], jj[i-2], jj[i-1] == jj[i-2] - 1)
            if i > 2:
                if ((self.codes[j][i-1].code_no == self.codes[j][i-2].code_no + sign)&((self.codes[j][i-1].code_no_status > 0)|(self.codes[j][i-2].code_no_status > 0))):    ## extrapolate from last two
                    #print(mm, jj[i-1], jj[i-2], jj[i-1] == jj[i-2] - 1)
                    self.codes[j][i].code_no = self.codes[j][i-1].code_no + sign
                    self.codes[j][i].code_no_status = 0
                    continue

                if mm == self.codes[j][i-2].code_no + 2*sign:                 ## just one event was off
                    self.codes[j][i].code_no = mm
                    self.codes[j][i].code_no_status = 1
                    self.codes[j][i-1].code_no = mm - sign
                    continue

                self.codes[j][i].code_no = mm                         ## otherwise use new value
                if self.codes[j][i-1].code_no + sign == mm:
                    self.codes[j][i].code_no_status = 1
                else:
                    self.codes[j][i].code_no_status = 0
            else:
                self.codes[j][i].code_no = gm2.np.nan
                self.codes[j][i].code_no_status = 0
          else:
            self.codes[j][i].code_no = gm2.np.nan
            self.codes[j][i].code_no_status = 0
        self.hasNums = True

    def interpolate(self, j, spread=2, plot=False):
        if self.verbosity > 0:
            print("interpolate codes")
        sign = 1 if self.cw else -1
        ok = []
        out = []
        #for j in range(2):
        locked = False
        inter_n = 0
        lastlock   = -1
        lastlock_i = -1
        for i in range(len(self.codes[j])):
           nn = self.codes[j][i].code_no
           out.append(nn)
           ok.append(False)

           if i > 2:
              if ((self.codes[j][i].code_no == self.codes[j][i-1].code_no + sign)&
                  (self.codes[j][i].code_no == self.codes[j][i-2].code_no + 2*sign)):## 3 in a row -> lock to it
                  if ((lastlock==-1) | 
                      ((abs(lastlock - self.codes[j][i].code_no) < (i - lastlock_i)*spread))):
                      # NEW LOCK
                      locked  = True
                      ok[i]   = True
                      ok[i-1] = True
                      #print(lastlock - jj[i])
                      lastlock = self.codes[j][i].code_no
                      lastlock_i = i
                      if i > 3:
                          if out[i-2] != out[i-3]:
                              ok[i-2] = True
                      else:
                          ok[i-2] = True
                      # if we have been interpolating, check if it fits or revert
                      if inter_n > 0:
                          if nn != out[i-1] + sign: # interpolation is not ok
                              for j_ in gm2.np.arange(3, inter_n+1):
                                  ok[i-j_] = False
                              out[i-1] = self.codes[j][i-1].code_no
                              out[i-2] = self.codes[j][i-2].code_no
                          inter_n = 0
                      continue

           if i > 1:                 # two times same value
               if self.codes[j][i].code_no == self.codes[j][i-1].code_no: 
                   ok[i-1] = False
                   ok[i]   = False
                   continue

           # if we reach here, the new value is off
           if (locked | (inter_n > 0)): # if previous was good or we are interpolating -> interpolate 
               out[i] = out[i-1] + sign
               inter_n += 1
               ok[i] = True

           locked = False


        ok  = gm2.np.array(ok)
        out = gm2.np.array(out)
        if plot:
            for j in range(2):
              out[ok==False] = gm2.np.nan
            gm2.plt.plot(self.out)
            gm2.plt.show()
        return out, ok


    #def print(self, j=0):
    #     for i, nn in enumerate(self.num[j]):
    #          print(i, nn, self.out[j][i], self.ok[j][i])


    def loadPos(self, interpolate=None, plot=0, drop=0):
        # drop 1: add nan at bad codes 
        # requires code_no
        if not self.hasNums:
            self.Code2Num()
        sign = 1 if self.cw else -1
        self.pos_time = [[],[]]
        self.pos_no   = [[],[]]  
        for j in range(2):
            if interpolate is None:
                out = gm2.np.array([code.code_no for code in self.codes[j]])
                ok  = gm2.np.array([code.code_no_status == 0 for code in self.codes[j]])
            else:
                out, ok = self.interpolate(j, interpolate)
            #print(len(ok), len(self.codes[j]), ok)
            for i in range(len(self.codes[j])):
                code = self.codes[j][i].code_num
                if ((self.codes[j][i].status == 0) & # 30 clks
                    (i>0)):
                    if ok[i-1] == True:
                        # assume ok between last abs marker and here
                        for clk_n in gm2.np.arange(30):
                            clk_index = self.codes[j][i-1].clk_n + clk_n
                            self.pos_time[j].append(self.time[self.clk_index[j][clk_index]])
                            self.pos_no[j].append(out[i-1] + sign*clk_n/30.0)
                    else:
                        if drop > 0:
                            clk_index = self.codes[j][i-1].clk_n
                            self.pos_time[j].append(self.time[self.clk_index[j][clk_index]])
                            self.pos_no[j].append(gm2.np.nan)
                elif self.codes[j][i].status == 2:
                    if ((ok[i-1] == True)&(drop<=1)):
                        clk_index = self.codes[j][i-1].clk_n
                        time_ = self.time[self.clk_index[j][clk_index]]
                        self.pos_time[j].append(time_)
                        self.pos_no[j].append(out[i-1])
                        if drop > 0:
                            self.pos_time[j].append(time_+1) # right after
                            self.pos_no[j].append(gm2.np.nan)
                    else:
                        if drop > 0:
                            clk_index = self.codes[j][i-1].clk_n
                            self.pos_time[j].append(self.time[self.clk_index[j][clk_index]])
                            self.pos_no[j].append(gm2.np.nan)
                else:
                    if drop > 0:
                        clk_index = self.codes[j][i-1].clk_n
                        self.pos_time[j].append(self.time[self.clk_index[j][clk_index]])
                        self.pos_no[j].append(gm2.np.nan)


        self.pos_time = [gm2.np.array(self.pos_time[0]), gm2.np.array(self.pos_time[1])]
        self.pos_no   = [gm2.np.array(self.pos_no[0]),   gm2.np.array(self.pos_no[1])]

        if plot > 0:
            t0 = self.val_times[0].min()
            gm2.plt.plot((self.pos_time[0]-t0)/1e9, self.pos_no[0], '.')
            gm2.plt.plot((self.pos_time[1]-t0)/1e9, self.pos_no[1], '.')
            gm2.plt.show()

        if plot > 1:
            tt = gm2.util.interp1d(self.pos_time[0], self.pos_no[0])
            gm2.plt.hist(self.pos_no[1][100:-100] - tt(self.pos_time[1][100:-100]), bins=gm2.np.arange(1.83,1.9,0.0001), histtype='stepfilled')

    def plotAtPos(self, pos, plotPos = False, dt=10):
        idx = gm2.util.nearestIndex(self.tr_pos[:,0], pos + 0.4)
        if plotPos:
            return self.plotAtTime(self.tr_time[idx,0], pos = pos, dt=dt)
        else:
            return self.plotAtTime(self.tr_time[idx,0], dt=dt)

    def plotAtTime(self, time, dt=10.0, lables=True, figure=True, pos=False):
        if pos:
           posAt = gm2.util.interp1d(self.tr_time[:,0], self.tr_pos[:,0], fill_value='extrapolate')

        if lables:
            if not self.hasNums:
                self.Code2Num()
        else: 
            if not hasattr(self, 'clk_index'):
                self.loadClkIndex()
                self.loadCodes()
        t0 = self.time.min() 
        t_start = time
        t_end   = t_start + dt * 1e9 
        s_raw   = ((self.time>t_start) & (self.time<t_end)) # selection on raw waveforms
        s_index = [((self.time[self.clk_index[0]] > t_start ) & (self.time[self.clk_index[0]] < t_end)), # selection on clks
                   ((self.time[self.clk_index[1]] > t_start ) & (self.time[self.clk_index[1]] < t_end))]

        code_indeces = [gm2.np.array([c.clk_n for c in self.codes[0]]), # used to find codes inside our solution
                        gm2.np.array([c.clk_n for c in self.codes[1]])]

        s_codes = []
        for i in range(2):
            if s_index[i].shape[-1] > 0:
                s_codes.append(((code_indeces[i] > gm2.np.argwhere(s_index[i]).min() - 12 ) &  (code_indeces[i] < gm2.np.argwhere(s_index[i]).max() )))
            else:# selection of codes
                s_codes.append(gm2.np.array([]))

        if figure:
            fig = gm2.plt.figure(figsize=[gm2.plotutil.figsize()[0]*2.0, gm2.plotutil.figsize()[1]*1.5])
        axs = []
        for i in range(2):
            if i > 0:
                axs.append(gm2.plt.subplot(411+2*i, sharex=axs[0]))
            else:
                axs.append(gm2.plt.subplot(411+2*i))
            if pos:
                j = i
                gm2.plt.plot(posAt(self.time[s_raw][:-1]), self.clk[j][s_raw][:-1], color=gm2.colors[i], alpha=0.5) 
                #gm2.plt.plot(posAt(self.time[s_raw][:-1]), self.clk[2+j][s_raw][:-1], color=gm2.colors[i], alpha=0.5) 
                gm2.plt.plot(posAt(self.time[self.clk_index[i]][s_index[i]]), self.clk[i][self.clk_index[i]][s_index[i]], '.', color= gm2.colors[i])
            else:
                gm2.plt.plot((self.time[s_raw][:-1]-t0)/1e9, self.clk[i][s_raw][:-1], color=gm2.colors[i], alpha=0.5) 
                gm2.plt.plot((self.time[self.clk_index[i]][s_index[i]] - t0)/1e9, self.clk[i][self.clk_index[i]][s_index[i]], '.', color= gm2.colors[i])
            if lables:
                try:
                    code_i = gm2.np.argwhere(s_codes[i]).min()
                    if code_i > 0:
                        code_i -= 1
                    n_ = 0
                    while (code_indeces[i][code_i]+n_ < gm2.np.argwhere(s_index[i]).max()):
                        n_ += 1
                        text_x = (self.time[self.clk_index[i][code_indeces[i][code_i]+n_]]-t0)/1e9
                        text_y = self.clk[i][self.clk_index[i]][s_index[i]].max()
                        if text_x > (t_start-t0)/1e9:
                            gm2.plt.gca().text(text_x, text_y, str(n_),
                                horizontalalignment='center',
                                verticalalignment='bottom', size='xx-small')
                        if code_i < len(code_indeces[i])-1:
                          if code_indeces[i][code_i]+n_ >= code_indeces[i][code_i+1]:
                            code_i += 1
                            n_ = 0
                except:
                    pass
            gm2.plt.setp(gm2.plt.gca().get_xticklabels(), visible=False)
            gm2.plt.ylabel("clk")

            axs.append(gm2.plt.subplot(412+2*i, sharex=axs[0]))
            if pos:
                pass
                gm2.plt.plot(posAt(self.time[s_raw][1:-1]), self.abs[i][s_raw][1:-1], color=gm2.colors[i], alpha=0.5) 
                gm2.plt.plot(posAt(self.time[self.clk_index[i]][s_index[i]]), self.abs[i][self.clk_index[i]][s_index[i]], '.', color= gm2.colors[i])
            else:
                gm2.plt.plot((self.time[s_raw][:-1]-t0)/1e9, self.abs[i][s_raw][:-1], color=gm2.colors[i], alpha=0.5) 
                gm2.plt.plot((self.time[self.clk_index[i]][s_index[i]] - t0)/1e9, self.abs[i][self.clk_index[i]][s_index[i]], '.', color= gm2.colors[i])
            if s_codes[i].shape[-1]>0:
                for j, c in enumerate(code_indeces[i][s_codes[i]]): # c is the index
                    if lables:
                        if j == 0:
                            text_x = (gm2.np.max([self.time[self.clk_index[i][c]], self.time[self.clk_index[i][gm2.np.argwhere(s_index[i]).min()]]])-t0)/1e9
                        else:
                            text_x = (self.time[self.clk_index[i][c]]-t0)/1e9
                        text_y = gm2.np.max( self.abs[i][self.clk_index[i]][s_index[i]])
                        code_ = self.codes[i][gm2.np.argwhere(code_indeces[i]==c)[0][0]]
                        pattern_ = ''.join(map(str, (code_.code_a < (code_.code_a.min() + code_.code_a.max())/2.0).astype(int)[:-1]))
                        gm2.plt.gca().text(text_x, text_y, pattern_+(": %.0f" % code_.code_num),
                            horizontalalignment='left',
                            verticalalignment='baseline')
                            #rotation=0)
                            #transform=plt.gca().transAxes)
                    for jj in range(12):
                        index_ = c+jj
                        if (index_ >= gm2.np.argwhere(s_index[i]).min()) & (index_ <= gm2.np.argwhere(s_index[i]).max()):
                            if pos:
                                gm2.plt.plot(posAt(self.time[self.clk_index[i][index_]]), self.abs[i][self.clk_index[i][index_]], 'o', color= gm2.colors[2+i])
                            else:
                                gm2.plt.plot((self.time[self.clk_index[i][index_]] - t0)/1e9, self.abs[i][self.clk_index[i][index_]], 'o', color= gm2.colors[2+i])
            if i == 0:     
                gm2.plt.setp(gm2.plt.gca().get_xticklabels(), visible=False)
            gm2.plt.ylabel("abs")

        if pos:
            gm2.plt.xlabel("position [deg]")
        else:
            gm2.plt.xlabel("time [s]")
        gm2.plt.subplots_adjust(hspace=0)
        gm2.despine()
        if figure:
            return fig
        #gm2.plt.show()

'''
import gm2
from gm2 import plt, np
bc = gm2.Barcode([3997]) 
bc.loadPos(interpolate=2000000000000, drop=2)  
t0 = bc.time.min()

inter = gm2.util.interp1d(bc.pos_time[1], bc.pos_no[1], fill_value='extrapolate')

plt.plot((bc.pos_time[1]- bc.time.min())/1e9, bc.pos_no[1], '.', markersize=4, label="")
plt.plot((bc.pos_time[1]- bc.time.min())/1e9, inter(bc.pos_time[1]), '-', markersize=2, label="")
plt.plot((bc.pos_time[0]- bc.time.min())/1e9, inter(bc.pos_time[0]), '.:', markersize=2, label="")
plt.show()


ax = plt.subplot(211)
plt.plot( (bc.pos_time[1]- bc.time.min())/1e9, bc.pos_no[1], '.', markersize=2, label="")
plt.plot( (bc.pos_time[0]- bc.time.min())/1e9, bc.pos_no[0], '.', markersize=2, label="")
plt.ylabel("position [abs space]")
plt.setp(plt.gca().get_xticklabels(), visible=False)

plt.subplot(212, sharex=ax)
plt.plot( (bc.pos_time[0]- bc.time.min())/1e9, (bc.pos_no[0]-inter(bc.pos_time[0]))*30.0, '.', markersize=2)
plt.ylabel("distance [clk space]")
plt.ylim([55, 57])
plt.xlabel("time [s]")

gm2.despine()
plt.show()


b = [[  0, 55],
     [ 60, 115],
     [120, 175],
     [179, 234],
     [239, 294],
     [299, 354],
     [359, 414],
     [419, 474],
     [479, 533],
     [538, 594],
     [598, 653],
     [658, 714],
]

#plt.plot( (bc.pos_time[1]- bc.time.min())/1e9, (bc.pos_no[1]-inter(bc.pos_time[1]))*30.0, '.', markersize=2)
plt.plot( bc.pos_no[0]*60, (bc.pos_no[0]-inter(bc.pos_time[0]))*60.0 -112, '.', markersize=2)
for bb in b:
    plt.plot([bb[0]*60., bb[0]*60.],[1, 3], ':', color='black', alpha=0.4, linewidth=1)
    plt.plot([bb[1]*60., bb[1]*60.],[1, 3], ':', color='black', alpha=0.4,linewidth=1)

plt.ylabel("distance [clk space]")
plt.ylim([-1, 1])
plt.xlabel("distance [mm]")

gm2.despine()
plt.show()


'''

'''
A = np.zeros([])
for i in range(bc.pos_time[1]):
    if not np.isnan(bc.pos_no[1][i]):
        if not np.isnan(inter(bc.pos_time[1][i]))
            d_ = (bc.pos_no[1]-inter(bc.pos_time[1]))*30.0
            full_ = np.floor(d_)
            fraction_ = d_ - full_
            [0][1]*full_ + [fraction_]
'''




