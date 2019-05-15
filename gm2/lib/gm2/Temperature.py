from ROOT import TFile
import os
import numpy as np
import datetime
import gm2

#import sys
#import numpy as np
#import gm2
#if 'GM2' not in os.environ:
#    raise Exception('The environment variable GM2 is not set.')
if 'ARTTFSDIR' not in os.environ:
    raise Exception('The environment variable ARTTFSDIR is not set.')



class Temperature:
    def __init__(self, update = False, db=False):
        """
        Access temperature information.

        Args:
            db (bool) : use online db if true, otherwise nightely aggregated root files. Defaults to false.
        """
        if update:
            self.update()
        self.data = None
        if db:
            import gm2.OnlineDB
            self.db = gm2.OnlineDB()
            self.file = None
        else:
            self.db = None
            self.file = TFile(os.environ['ARTTFSDIR']+"../SlowControlData/temp_plots.root")
            # https://muon.npl.washington.edu/elog/g2/General+Field+Team/1197
            self.mscb = {'A':'323', 'B':'13e','C':'323', 'D':'13e', 'E':'323', 'F':'13e', 'G':'323', 'H':'13e', 'I':'323', 'J':'13e', 'K':'323', 'L':'13e'} 
            self.P    = {'A':{'Top':'1_0', 'Back':'1_1', 'Bottom':'1_2','Air':'1_3'},
                         'B':{'Top':'1_0', 'Back':'1_1', 'Bottom':'1_2','Air':'1_3'},
                         'C':{'Top':'1_4', 'Back':'1_5', 'Bottom':'1_6','Air':'1_7'},
                         'D':{'Top':'1_4', 'Back':'1_5', 'Bottom':'1_6','Air':'1_7'},
                         'E':{'Top':'2_0', 'Back':'2_1', 'Bottom':'2_2','Air':'2_3'},
                         'F':{'Top':'4_6', 'Back':'2_1', 'Bottom':'2_0','Air':'2_3'},
                         'G':{'Top':'2_4', 'Back':'2_5', 'Bottom':'2_6','Air':'2_7'},
                         'H':{'Top':'2_4', 'Back':'2_5', 'Bottom':'2_6','Air':'2_7'},
                         'I':{'Top':'3_0', 'Back':'3_1', 'Bottom':'3_2','Air':'4_3'},
                         'J':{'Top':'3_7', 'Back':'3_1', 'Bottom':'3_2','Air':'3_3'},
                         'K':{'Top':'3_4', 'Back':'3_5', 'Bottom':'3_6','Air':'3_7'},
                         'L':{'Top':'4_1', 'Back':'4_2', 'Bottom':'3_5','Air':'4_4'}} 

    def update(self):
        print("run 'gm2/scripts/updateTemp.sh' manually")

    def get(self, yoke, pos):
        if self.data is None: 
            if yoke not in self.mscb:
                raise ValueError("Yoke '"+yoke+"' is not within 'A' - 'L'")
            if pos not in self.P['A']:
                raise ValueError("Position '"+pos+"' is one of ['Top', 'Back', 'Bottom', 'Air'")
        else:
            if yoke not in self.data.keys():
                raise ValueError("Yoke '"+yoke+"' is not present"+ self.data.keys())
            if pos not in self.data[yoke].keys():
                raise ValueError("Position '"+pos+"' is not present in "+self.data[yoke].keys())

        if self.data is None:
            g = self.file.Get("magnet"+yoke+"_mscb"+self.mscb[yoke]+"_Temp_P"+self.P[yoke][pos])
            if g:
                return np.frombuffer(g.GetX()) *1e9, np.frombuffer(g.GetY())
            else:
                return np.array([]), np.array([])
        else:
             return np.array(self.data[yoke][pos]['times']), np.array(self.data[yoke][pos]['values'])

    def load(self, start = None, end = None):
        import gm2.util
        if self.db is None:
            raise NameError('load is only present if the class uses the onlineDB')
        if start is None:
            if end is None:
                end = gm2.util.datetime.now()
            start = end - gm2.util.timedelta(hours=12)
        if end is None:
            end = gm2.util.datetime.now()
        self.data = self.db.getTemperature(gm2.util.datetime2ts_dt(start), 
                                           gm2.util.datetime2ts_dt(end))
        self.start = start
        self.end   = end


    def plotAll(self,  marker='.', start = None, end =None, ylim=False):
        from gm2 import plt
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
        f = gm2.plt.figure(figsize=figsize)
        ax1 = gm2.plt.subplot(311)
        yokes = [chr(i) for i in range(ord('A'), ord('A')+12)]
        self.plot(yokes=yokes, posistions=['Top'], rel=True, show=False, marker='.', alpha=0.5, start=start, end=end)
        self.plot(yokes=yokes, posistions=['Bottom'], rel=True, show=False, marker='.', alpha=0.5, start=start, end=end)
        xlim = ax1.get_xlim()
        gm2.plt.plot(xlim, [0.0,0.0] ,'--', linewidth=1, color='black', alpha=0.5)
        ax1.set_xlim(xlim)
        ax1.set_ylabel("temp drift [C]")
        ax1.set_title("yokes")
        ax3 = ax1.twinx() 
        ax1.set_xlabel("")
        ax1.set_ylim([-0.1,0.1])
        ax3.set_ylim([0,30])
        self.plot(yokes=['Outside'],   posistions=['ES&H'], rel=False, show=False, marker='o', start=start, end=end)
        self.plot(yokes=['NorthWall'], posistions=['MiddleWest'], rel=False, show=False, marker='^', start=start, end=end)
        ax3.set_ylabel("temp [C]")
        plt.legend()
        ax2 = gm2.plt.subplot(312, sharex=ax1)
        ax2.set_title("vacuum chambers")
        self.plot(yokes=yokes, posistions=['Vac'], rel=True, show=False, marker='x', markersize=4, alpha=0.5, start=start, end=end) 
        ax2.set_ylabel("temp drift [C]")
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_xlabel("") 
        gm2.plt.plot(xlim, [0.0,0.0] ,'--', linewidth=1, color='black', alpha=0.5)
        ax2.set_xlim(xlim)
        ax4 = gm2.plt.subplot(313, sharex=ax1)
        self.plotGradient(yokes=yokes, rel=True, show=False, start=start, end=end)
        ax4.set_ylabel("temp drift [C]")
        ax4.set_title("yoke gradients (top-bottom)")
        gm2.plt.plot(xlim, [0.0,0.0] ,'--', linewidth=1, color='black', alpha=0.5)
        ax4.set_xlim(xlim)
        ax4.set_xlabel("")
        gm2.plt.setp(ax1.get_xticklabels(), visible=False)
        gm2.plt.setp(ax2.get_xticklabels(), visible=False)
        gm2.plt.setp(ax3.get_xticklabels(), visible=False)
        gm2.despine()
        #ax4.set_ylim([])
        return f 


    def plot(self, yokes, posistions, marker='.', markersize=2, alpha=1.0, start = None, end =None, rel=False, ylim=False, show=True):
        import gm2.plotsettings
        import gm2.util
        if not self.db is None:
            if self.data is None:
                self.load(start, end)

        for yoke in yokes:
            for pos in posistions:
                time, temp = self.get(yoke, pos)
                #time_ = np.vectorize(datetime.datetime.fromtimestamp)(time)
                s = np.full([time.shape[0]], 1) == 1
                if self.db is None:
                    if not start is None:
                        s = s&(time[:] > gm2.util.datetime2ts(*start))
                    if not end is None:
                        s = s&(time[:] < gm2.util.datetime2ts(*end)) 

                #gm2.plotsettings.plt.plot_date(time_, temp, marker, label="Yoke "+yoke+", "+pos, markersize=2)
                if rel:
                    gm2.plotutil.plot_ts(time[s], temp[s] - temp[s][:].mean(), marker, label="Yoke "+yoke+", "+pos, markersize=markersize, alpha=alpha)
                else:
                   gm2.plotutil.plot_ts(time[s], temp[s], marker, label=""+yoke+", "+pos, markersize=markersize, alpha=alpha)
        gm2.plotsettings.plt.xlabel("date")
        if not ylim is None:
            gm2.plt.ylim(ylim)
        if rel:
            gm2.plotsettings.plt.ylabel("temperature change [C]")
        else:
            gm2.plotsettings.plt.ylabel("temperature [C]")
        if show:
            gm2.plotsettings.plt.legend(markerscale=4)
            gm2.plotsettings.despine()
            gm2.plotsettings.plt.show()


    def plotGradient(self, yokes, marker='.',start = None, end =None, rel=False, ylim=None, show=True):
        import gm2.util 
        if not self.db is None:
            if self.data is None:
                self.load(start, end)

        for yoke in yokes:
            time_t, temp_t = self.get(yoke, 'Top')
            time_b, temp_b = self.get(yoke, 'Bottom')
            temp_t_f = gm2.util.interp1d(time_t, temp_t, fill_value='extrapolate')
            s = np.full([time_b.shape[0]], 1) == 1
            if self.db == None:
                if not start is None:
                    s = s&(time_b[:] > gm2.util.datetime2ts(*start))
                if not end is None:
                    s = s&(time_b[:] < gm2.util.datetime2ts(*end))
            dt = temp_t_f(time_b[s]) - temp_b[s]
            if rel:
                gm2.plotutil.plot_ts(time_b[s], dt - dt[:].mean(), marker, label="Yoke "+yoke, markersize=2)
            else:
                gm2.plotutil.plot_ts(time_b[s], dt, marker, label="Yoke "+yoke, markersize=2)
        gm2.plotsettings.plt.xlabel("date")
        if not ylim is None:
            gm2.plt.ylim(ylim)
        if rel:
           gm2.plotsettings.plt.ylabel("changin in \n temperature gradient\n top - bottom [C]")
        else:
           gm2.plotsettings.plt.ylabel("temperature gradient\n top - bottom [C]")
        if show:
            gm2.plotsettings.despine()
            gm2.plotsettings.plt.legend(markerscale=4)
            gm2.plotsettings.plt.show()



