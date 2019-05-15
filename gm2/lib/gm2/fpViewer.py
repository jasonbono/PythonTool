#!/usr/bin/env python
import gm2
from gm2.plotutil import plot_ts

class fpViewer(object):
    """Class to bundle fixed probe plots utilities.
    
    Attributes:
        fp(gm2.FixedProbe) : fixed probe class.
        
    """
    def __init__(self, runs, end = None, prefix = None):
        """fpViewer constructor. Initialized with a run list or first run number and potentially end run number.

        Args:
            runs (list[int], optional) : run numbers. or first run number.
            end (int, optional) :  if runs is a single number a end run number can be specified
        """
        self.prefix = prefix
        if type(runs) is int:
           if end is None:
               self.runs = [runs]
           else:
               self.runs = gm2.np.arange(runs, end+1)
        else:
            self.runs = runs
        self.fp = gm2.FixedProbe(self.runs, True, prefix=prefix)

        self.phi = self.fp.phi
        self.i = None # Issues


    def selectStation(self,yoke, azi):
        return (self.fp.id['yoke'] == yoke)&(self.fp.id['azi'] == azi)

    def selectProbe(self,rad, layer):
        return (self.fp.id['rad'] == rad)&(self.fp.id['layer'] == layer)

    def mkdir(self, folder, dirname = None): 
        # create folder for pngs
        import os
        try:
            os.mkdir(folder)
        except:
            pass
        if not dirname is None:
            try:
                os.mkdir(folder+"/"+dirname)
            except:
                pass
    

    def jumpAt(self, year, month, day, hour, mins, secs, tDelta = 2, tOffset = 3):
        jumpt_t = gm2.util.datetime(year, month, day, hour, mins, secs)

        return self.jump(jumpt_t)

    def jumps(self, dts, tDelta = 2, tOffset = 3): 
        
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2, gm2.plt.rcParams['figure.figsize'][1] * 1]
        gm2.plt.figure(figsize=figsize)
        for dt in dts:
            self.jump(dt, multiple = True, tDelta = tDelta, tOffset = tOffset)
       
        ylim = gm2.plt.gca().get_ylim()
        ylim  = [ylim[0]*0.8, ylim[1]*0.8]
        for nn, j in enumerate(gm2.np.arange(22.5, 22.5 + 360, 360/8.)):
            if nn  in [0]:
                gm2.plt.plot([j,j], [ylim[0], ylim[1]], '-', alpha=0.5, color=gm2.sns.color_palette()[5], label="upper radial stop")
                gm2.plt.plot([j+15,j+15], [ylim[0], ylim[1]], '-', alpha=0.5, color=gm2.sns.color_palette()[6],  label="lower radial stop")
            else:
                gm2.plt.plot([j,j], [ylim[0], ylim[1]], '-', alpha=0.5, color=gm2.sns.color_palette()[5])
                gm2.plt.plot([j+15,j+15], [ylim[0], ylim[1]], '-', alpha=0.5, color=gm2.sns.color_palette()[6])
        gm2.plt.plot([0,360], [0,0], '--', color='black', alpha=0.8, linewidth=0.5)
        gm2.plt.xlabel("azimuth [deg]")
        gm2.plt.ylabel("field jump [ppm]")
        gm2.plt.legend(markerscale=4, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
        gm2.plt.subplots_adjust(top=0.8)
        gm2.sns.despine()
        gm2.plt.show()
        gm2.plt.close('all')
        #plt.savefig("plots/fpJump_"+str(year)+str(month)+str(day)+"_"+str(hour)+str(mins)+".png")

    def jump(self, dt, tDelta = 2, tOffset = 3, multiple = False, show = False):
        import datetime
        fp_phi = self.fp.getPhi()
        fp_time = self.fp.time
        fp_freq = self.fp.freq

        dtHalf  = datetime.timedelta(minutes = tDelta/2.)
        dtOffset = datetime.timedelta(minutes = tOffset/2.) 
        jumpt_t = dt

        def timestamp(dt):
            return gm2.util.datetime2ts_dt(dt)
        s_before = (fp_time[:,0] > timestamp(jumpt_t - dtOffset - dtHalf))&(fp_time[:,0] < timestamp(jumpt_t - dtOffset + dtHalf))
        s_after =  (fp_time[:,0] > timestamp(jumpt_t + dtOffset - dtHalf))&(fp_time[:,0] < timestamp(jumpt_t + dtOffset + dtHalf))
        f_before = gm2.np.nanmean(self.fp.freq[s_before,:], axis=0)
        f_after  = gm2.np.nanmean(self.fp.freq[s_after,:], axis=0)


        yokes_ = gm2.np.arange(ord('A'), ord('L')+1)
        aziIds = gm2.np.arange(1,6+1)
        phi = fp_phi
        #for yoke_ in yokes_:
        #    yoke = chr(yoke_)
        #    for aziId in aziIds:
        #        s_station = self.selectStation(yoke, aziId)
        #        phi.append(fp_phi[s_station][0])
        #phi = gm2.np.array(phi)
       
        #phi = (gm2.np.array([ord(i) for i in self.fp.id['yoke']]) - 65) * 30 - 15 + (self.fp.id['azi']-1) * 30/6. + 30/12.
        #print phi
        #print f_before.shape


        layers = ["T", "B"]
        radIds = ["O", "M", "I"]

        if multiple:
            s = self.selectProbe([radIds[1]], [layers[1]])
            order = gm2.np.argsort(phi[s])
            gm2.plt.plot(phi[s][order], (f_after[s]-f_before[s])[order]*gm2.HZ2PPM, '.-', markersize=2, label=dt.strftime("%m/%d/%Y %H:%M"))
        else:
            for radId in radIds:
                for layer in layers:
                    s = self.selectProbe(radId, layer)
                    gm2.plt.plot(phi[s], (f_after[s]-f_before[s])*gm2.HZ2PPM, '.', markersize=2, label=layer+"-"+radId)
        if not multiple:
           gm2.plt.xlabel("azimuth [deg]")
           gm2.plt.ylabel("field jump [ppm]")
           gm2.plt.legend(markerscale=4)
           gm2.plt.title("Run "+str(self.runs[0])+" "+dt.strftime("%m/%d/%Y %H:%M"))
           gm2.despine()
           gm2.plt.savefig("plots/fpJump_"+dt.strftime("%m%d%Y_%H%M")+".png")
           if show: 
              gm2.plt.show()

    def addIssues(self, y=0):
        import gm2
        system_names  = []
        if self.i is None:
            self.i = gm2.Issues(self.runs, prefix=self.prefix)
        s = (self.i.time > 0)&(self.i.system>0)
        markers = ['^','x','d','s']
        for system in set(self.i.system[s]):
          for type_ in set(self.i.typ[s]):
            s_ = s&(self.i.system == system)&(self.i.typ==type_)
            #print int(system), int(type_)
            if type_ > 5:
                issue_name = "Issue "+self.i.system_names_short[int(system)]+" "+str(int(type_))
            else:
                issue_name = "Issue "+self.i.system_names_short[int(system)]+" "+self.i.type_name[int(system)][int(type_)]
            gm2.plotutil.plot_ts(self.i.time[s_], [y]*self.i.time[s_].shape[0], markers[int(type_)%4], markersize=2, color=gm2.sns.color_palette()[((int(system)-1)+int(type_))%10], label=issue_name)


    def azMean(self, show=False, folder = "plots", ylim=0.6, ylim2=0.3, showSel=True, showIssues=True):
        if showSel:
           fb = gm2.Feedback(self.runs)
           sel = fb.probeList()
        s_top    = (self.fp.id['layer'] == 'T')
        s_bot    = (self.fp.id['layer'] == 'B')

        skip = 1

        fp_time = self.fp.time
        fp_freq = self.fp.freq
        d = fp_freq.mean(axis=1) * gm2.HZ2PPM
        if showSel:#not gm2.np.isnan(sel):
          d_sel =fp_freq[:,sel].mean(axis=1) * gm2.HZ2PPM
        sq = (fp_freq[:, s_top].mean(axis=1) - fp_freq[:, s_bot].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.y[0])*45.
        nq = gm2.np.zeros([fp_freq.shape[0]])
        yokes_ = gm2.np.arange(ord('A'), ord('L')+1)
        aziIds = gm2.np.arange(1,6+1)
        for yoke_ in yokes_:
            yoke = chr(yoke_)
            for aziId in aziIds:
                s_station = self.selectStation(yoke, aziId)
                s_inner  = (self.fp.id['rad']   == 'I')&s_station
                s_outer  = (self.fp.id['rad']   == 'O')&s_station
                s_center = (self.fp.id['rad']   == 'M')&s_station
                if (len(gm2.np.argwhere(s_inner)) > 0):
                  nq += (fp_freq[:, s_outer].mean(axis=1) - fp_freq[:, s_inner].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.x[2])*45.
                else:
                  nq += (fp_freq[:, s_outer].mean(axis=1) - fp_freq[:, s_center].mean(axis=1)) * gm2.HZ2PPM/(gm2.FP.probes.position.x[2])*45.

        nq /= (yokes_.shape[-1] * aziIds.shape[-1])

        dirname = str(self.runs[0])+"to"+str(self.runs[-1])
        prefix = "fpMean"
        self.mkdir(folder)
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
        
        probe_t = 171
        f = gm2.plt.figure(figsize=figsize)
        ax1 = gm2.plt.subplot(211)
        s_t = fp_time[skip:, probe_t] > 0
        plot_ts(fp_time[skip:, probe_t][s_t], d[skip:][s_t] - d[skip:][s_t][:].mean(),     '.', markersize=2, label="dipole", color=gm2.sns.color_palette()[0])
        if showSel:#not gm2.np.isnan(sel):
           plot_ts(fp_time[skip:, probe_t][s_t], d_sel[skip:][s_t] - d_sel[skip:][s_t][:].mean(), '.', markersize=2, label="dipole, feedback", color=gm2.sns.color_palette()[3])
        ax1.set_ylabel("dipole change [ppm]")
        ax1.set_ylim([-ylim, ylim])
        if showIssues:
            self.addIssues()
        ax2 = gm2.plt.subplot(212, sharex=ax1)

        plot_ts(fp_time[skip:, probe_t][s_t], nq[skip:][s_t]-nq[skip:][s_t][:].mean(), '.', markersize=2, label="norm quadrupole", color=gm2.sns.color_palette()[1])
        plot_ts(fp_time[skip:, probe_t][s_t], sq[skip:][s_t]-sq[skip:][s_t][:].mean(), '.', markersize=2, label="skew quadrupole", color=gm2.sns.color_palette()[2])

        ax2.set_ylim([-ylim2, ylim2])
        gm2.plt.xlabel("time")
        gm2.plt.ylabel('quadrupole change [ppm/45mm]')
        #gm2.plt.ylim([-1.0,1.0])
        gm2.despine()
        gm2.plt.subplot(211)
        gm2.plt.setp(ax1.get_xticklabels(), visible=False)
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles_ = handles + handles2
        labels_  = labels  + labels2
        lgnd = gm2.plt.legend(handles_, labels_, loc='upper center', markerscale=4, bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        #if showSel:#not gm2.np.isnan(sel):
        #   lgnd = gm2.plt.legend([handles[0], handles[1], handles2[0], handles2[1]],[labels[0], labels[1], labels2[0], labels2[1]], loc='upper center', markerscale=4, bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        #else:
        #   lgnd = gm2.plt.legend([handles[0], handles2[0], handles2[1]],[labels[0], labels2[0], labels2[1]], loc='upper center', markerscale=4, bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
        f.savefig(folder+"/"+prefix+"_"+dirname+".png")
        if show:
            gm2.plt.show()
        gm2.plt.close('all')
        #test = 1

        
    def getMp(self, yokes_, aziIds):
        mp = [[],[],[]]
        phi = []
        for yoke_ in yokes_:
            yoke = chr(yoke_)
            for aziId in aziIds:
                #print("Yoke "+yoke," : "+str(aziId))
                s_station = self.selectStation(yoke, aziId) 
                phi.append(self.phi[s_station][0])
                s_top    = (self.fp.id['layer'] == 'T')&s_station
                s_bot    = (self.fp.id['layer'] == 'B')&s_station
                s_inner  = (self.fp.id['rad']   == 'I')&s_station
                s_outer  = (self.fp.id['rad']   == 'O')&s_station
                s_center = (self.fp.id['rad']   == 'M')&s_station
                if (len(gm2.np.argwhere(s_inner)) > 0):
                  mp_n = (self.fp.freq[:, s_outer].mean(axis=1) - self.fp.freq[:, s_inner].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.x[2])*45.
                else:
                  mp_n = (self.fp.freq[:, s_outer].mean(axis=1) - self.fp.freq[:, s_center].mean(axis=1)) * gm2.HZ2PPM/(gm2.FP.probes.position.x[2])*45.
                   
                mp[0].append(self.fp.freq[:, s_station].mean(axis=1) * gm2.HZ2PPM)
                mp[1].append(mp_n)
                mp[2].append((self.fp.freq[:, s_top].mean(axis=1) - self.fp.freq[:, s_bot].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.y[0])*45.)
        for j in range(3):
            mp[j] = gm2.np.array(mp[j])
        return mp, phi


    def ring(self, gif = False, show = False, folder = "plots", dt_min = 10, rel = True, n = 1000,  ylim = 1.5, ylim2 = 0.2):

        yokes_ = gm2.np.arange(ord('A'), ord('L')+1)
        aziIds = gm2.np.arange(1,6+1) 
        fp_time = self.fp.time
        fp_freq = self.fp.freq

        prefix = "fpMpT"

        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
        
        mp, phi = self.getMp(yokes_, aziIds)

        dirname = str(self.runs[0])+"to"+str(self.runs[-1])
        self.mkdir(folder, dirname)

        probe_t = 171
        t_start = gm2.np.nanmin(fp_time[:, probe_t][fp_time[:, probe_t] > 0])
        t_end   = gm2.np.nanmax(fp_time[:, probe_t][fp_time[:, probe_t] > 0])


        dt = 1e9*60*dt_min
        t_sel_0 = t_start + dt
        s_t_0 = (fp_time[:, probe_t] > 0)&(fp_time[:, probe_t] > t_sel_0 - dt/2.)&(fp_time[:, probe_t] < t_sel_0 + dt/2.)
        for j in gm2.np.arange(1, (t_end-t_start)/dt):
            print ("Step", j, (dt_min*(j-1)), "min")
            t_sel = t_start + dt*(j-0.5)
            s_t = (fp_time[:, probe_t] > 0)&(fp_time[:, probe_t] > t_sel - dt/2.)&(fp_time[:, probe_t] < t_sel + dt/2.)
            print (gm2.util.ts2datetime(gm2.np.array([t_sel - dt/2.]))[0].strftime("%m/%d %H:%M:%S"), gm2.util.ts2datetime(gm2.np.array([t_sel + dt/2.]))[0].strftime("%m/%d %H:%M:%S"))

            f = gm2.plt.figure(figsize=figsize)
            ax1 = gm2.plt.subplot(211)
            if rel:
                gm2.plt.errorbar(phi, mp[0][:, s_t].mean(axis=1) - mp[0][:, s_t_0].mean(axis=1), yerr=mp[0][:, s_t].std(axis=1), label="dipole", fmt='x')
            else:
                gm2.plt.errorbar(phi, mp[0][:, s_t].mean(axis=1), yerr=mp[0][:, s_t].std(axis=1), label="dipole", fmt='x')
            ax1.set_ylabel("dipole [ppm]")
            if not rel:
                ax1.set_ylim([gm2.np.nanmin(mp[0]), gm2.np.nanmax(mp[0])])
            else:
                if (ylim > 0):
                    gm2.plt.ylim([-ylim, ylim])

            gm2.plt.title(gm2.util.ts2datetime(gm2.np.array([t_sel]))[0].strftime("%m/%d %H:%M:%S")+r' $\pm$ '+"%.1f min" % (dt_min/2.) + "(%.0f mins)" % (dt_min*(j-1)))
            ax2 = gm2.plt.subplot(212, sharex=ax1)
            if rel:
                gm2.plt.errorbar(phi, mp[1][:, s_t].mean(axis=1)-mp[1][:, s_t_0].mean(axis=1), yerr=mp[1][:, s_t].std(axis=1), label="norm quadrupole", fmt='x', color=gm2.sns.color_palette()[1])
                gm2.plt.errorbar(phi, mp[2][:, s_t].mean(axis=1)-mp[2][:, s_t_0].mean(axis=1), yerr=mp[2][:, s_t].std(axis=1), label="skew quadrupole", fmt='x', color=gm2.sns.color_palette()[2])
            else:
                gm2.plt.errorbar(phi, mp[1][:, s_t].mean(axis=1), yerr=mp[1][:, s_t].std(axis=1), label="norm quadrupole", fmt='x', color=gm2.sns.color_palette()[1])
                gm2.plt.errorbar(phi, mp[2][:, s_t].mean(axis=1), yerr=mp[2][:, s_t].std(axis=1), label="skew quadrupole", fmt='x', color=gm2.sns.color_palette()[2])

            ax2.set_xlabel("azimuth [deg]")
            ax2.set_ylabel("quadrupole [ppm/45mm]")
            if not rel:
                ax2.set_ylim([gm2.np.min([gm2.np.nanmin(mp[1]), gm2.np.nanmin(mp[2])]),
                              gm2.np.max([gm2.np.nanmax(mp[1]), gm2.np.nanmax(mp[2])])])
            else:
                if (ylim2 > 0):
                    gm2.plt.ylim([-ylim2, ylim2])
            gm2.despine()
            gm2.plt.subplot(211)
            gm2.plt.setp(ax1.get_xticklabels(), visible=False)
            #plt.gca().set_xticklabels([]);
            handles, labels = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            lgnd = gm2.plt.legend([handles[0], handles2[0], handles2[1]],[labels[0], labels2[0], labels2[1]], loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
            f.savefig(folder+"/"+dirname+"/"+prefix+"_%06i.png" % (dt_min*j))
            if show:
                gm2.plt.show()
            if j > n:
                break
            
        # combine plots to gif
        import os
        if gif:
            os.system("convert -delay 100 "+folder+"/"+dirname+"/"+prefix+"_*.png -loop 0 "+folder+"/"+prefix+"_"+dirname+".gif")
            print(dirname+"/"+prefix+"_"+dirname+".gif")
            


    def drift(self, show=True, ylim=5.0, folder="plots"):
        from scipy import signal
        drift = gm2.np.zeros([self.fp.n_probes])
        for p in range(self.fp.n_probes):
            s  = self.fp.time[:,p]>0
            mean = gm2.np.nanmedian(self.fp.freq[s,p])
            #bins = gm2.np.arange(gm2.np.nanmin(self.fp.freq[s,p])-mean, gm2.np.nanmax(self.fp.freq[s,p])-mean,3)
            bins = gm2.np.arange(-1000,1000,1)
            #try:
            popt, _ = gm2.plotutil.histWithGauss(gm2.plt.gca(), self.fp.freq[s,p]-mean, bins=bins)
            #except:
            #    popt = [0,0,100.]
            if p in []:
                gm2.plt.show()
            gm2.plt.clf()
            mean  += popt[1] 
            std   = popt[2]
            s2    = gm2.np.abs(self.fp.freq[s,p] - mean) < 5. * std
            #print p, mean, std, gm2.np.argwhere(s2).shape, self.fp.freq[s,p][s2].shape, self.fp.time[s,p][s2].shape
            wf_ = signal.savgol_filter(self.fp.freq[s,p][s2], 501, 3)
            drift[p] = gm2.np.nanmax(wf_) - gm2.np.nanmin(wf_)
            if p in []:
                gm2.plotutil.plot_ts(self.fp.time[s,p], self.fp.freq[s,p],'.')
                gm2.plotutil.plot_ts(self.fp.time[s,p][s2], self.fp.freq[s,p][s2],'.')
                gm2.plotutil.plot_ts(self.fp.time[s,p][s2], wf_,'.')
                gm2.plt.show()
             
        drift_mp = gm2.np.zeros([3, 72])
         
        yokes = gm2.np.arange(ord('A'), ord('L')+1)
        aziIds = gm2.np.arange(1,6+1) 
        mp, phi = self.getMp(yokes, aziIds)

        s  = self.fp.time[:,100]>0     
        for station in range(72):
            for mp_ in range(3):
                mean = gm2.np.nanmedian(mp[mp_][station,:])
                bins = gm2.np.arange(-10,10,0.005)
                #print station, mp_, mean, mp[mp_][station,s].mean()
                try:
                   popt, _ = gm2.plotutil.histWithGauss(gm2.plt.gca(), mp[mp_][station,s] - mean, bins=bins)
                except:
                   print("I am having problems to filter station %i" % station)
                   popt = [0,0.0,1.0]
                gm2.plt.clf()
                mean  += popt[1]
                std   = popt[2]
                #print station, mp_, mean, std
                s2    = gm2.np.abs(mp[mp_][station,s] - mean) < 5. * std
                try:
                    wf_   = signal.savgol_filter(mp[mp_][station,s][s2], 501, 3)
                    drift_mp[mp_, station] = gm2.np.nanmax(wf_) - gm2.np.nanmin(wf_)
                except:
                    drift_mp[mp_, station] = gm2.np.nan
        mp_names = ["dipole", "norm quad", "skew quad"] 
         
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.0] 
        f = gm2.plt.figure(figsize=figsize)
        for mp_ in range(3):
            gm2.plt.plot(phi, drift_mp[mp_,:],'x', label=mp_names[mp_], color=gm2.sns.color_palette()[mp_])

        for l in ['T', 'B']:
        
            for r in ['O','M','I']:
                s = (self.fp.id['layer']==l)&(self.fp.id['rad']==r)
                gm2.plt.plot(self.fp.phi[s], drift[s]*gm2.HZ2PPM,'.', label=l+r, alpha=0.3)

        gm2.plt.xlabel("azimuth [deg]")
        gm2.plt.ylabel("drift [ppm]")
        gm2.plt.ylim([0, ylim])
        gm2.plt.legend(ncol=3)
        gm2.despine()
        dirname = str(self.runs[0])+"to"+str(self.runs[-1])
        f.savefig(folder+"/fpDrift_"+dirname+".png")
        print(folder+"/fpDrift_"+dirname+".png")
        if  show:
            gm2.plt.show()
        gm2.plt.close('all')



    def view(self, mode=0, gif=False, folder="plots", show = False, yokes = None, aziId = None, ylim = 200., ylim2 = 50., ylim3=0.1, rel=True):
        """ Generates station wise fp overview 

            Args:
                gif(bool, optional) : generate gif (needs imagemagic on the system). Defaults False.
                mode(int, optional) : 0: freq, 1: power and length, 2: multipoles
                folder(string, optional) : output folder, Defaults 'plots'
                show(bool, optional) : show the plots. Defaults False,
                yokes(list(char), optional) : only selected yokes. Defaults None.
                aziId(list(int), optional) : only selcted aziIds. Defaults None.
                ylim (float, optional) : ylim of trend plots. -1 auto.
                rel (bool) : only for mode == 1. If rel is True relative fid power and length change is plotted. Defaults to True.
        """ 

        # Set default ylimes
        if ylim > 0:
            if mode == 1:
                ylim = 18
        if ylim2 > 0:
            if mode == 1:
                ylim2 = 8


        fp_time = self.fp.time
        fp_freq = self.fp.freq

        prefix = "fp"
        if mode == 1:
            prefix = "fpSig"
        if mode == 2:
            prefix = "fpMp"

        # if signal plots are requested, plot more data
        if mode == 1:
            #fp = gm2.FixedProbe([], False)
            #fp.fname_path    = "TreeGenFixedProbe/fixedProbe_DAQ"
            #fp.loadFiles(self.runs)
            fp_ = gm2.FixedProbe(self.runs, False, prefix=self.prefix)
            def callback():
               #return [self.fp.getTimeGPS(), self.fp.getAmplitude(), self.fp.getPower(), self.fp.getFidLength(), self.fp.getFidChi2()]
               return [fp_.getTimeGPS(), fp_.getPower(), fp_.getFidLength(), fp_.getFidChi2()]
            fp_time, fp_power, fp_length, fp_chi2  = fp_.loop(callback)

        skip = 1

        layers = ["T", "B"]
        radIds = ["O", "M", "I"]

        alpha = 1.0
        figsize = [gm2.plt.rcParams['figure.figsize'][1] * 2.0, gm2.plt.rcParams['figure.figsize'][1] * 1.5]
        from matplotlib.dates import DateFormatter
        formatter = DateFormatter('%m/%d\n%H:%M')


        dirname = str(self.runs[0])+"to"+str(self.runs[-1])
        self.mkdir(folder, dirname)

        test = 0
        if yokes is None:
           yokes_ = gm2.np.arange(ord('A'), ord('L')+1)
        else:
           yokes_ = [ord(y) for y in yokes]
        if aziId is None:
            aziIds = gm2.np.arange(1,6+1)
        else:
            aziIds = aziId

        for yoke_ in yokes_:
            yoke = chr(yoke_)
            for aziId in aziIds:
                print("Yoke "+yoke," : "+str(aziId))
                if test:
                    break
                s_station = self.selectStation(yoke, aziId) 
                phi_ = self.phi[s_station][0]

                f = gm2.plt.figure(figsize=figsize)
                if mode == 1:
                    ax1 = gm2.plt.subplot(311)
                else:
                    ax1 = gm2.plt.subplot(211)
                #print("Station", np.argwhere(s_station).shape)
                if mode == 2:
                    s_top    = (self.fp.id['layer'] == 'T')&s_station
                    s_bot    = (self.fp.id['layer'] == 'B')&s_station
                    s_inner  = (self.fp.id['rad']   == 'I')&s_station
                    s_outer  = (self.fp.id['rad']   == 'O')&s_station
                    s_center = (self.fp.id['rad']   == 'M')&s_station
                    if (len(gm2.np.argwhere(s_inner)) > 0):
                      mp_n = (fp_freq[:, s_outer].mean(axis=1) - fp_freq[:, s_inner].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.x[2])*45.
                    else:
                      mp_n = (fp_freq[:, s_outer].mean(axis=1) - fp_freq[:, s_center].mean(axis=1)) * gm2.HZ2PPM/(gm2.FP.probes.position.x[2])*45.
                       
                    mp = [fp_freq[:, s_station].mean(axis=1) * gm2.HZ2PPM, 
                         mp_n,
                         (fp_freq[:, s_top].mean(axis=1) - fp_freq[:, s_bot].mean(axis=1)) * gm2.HZ2PPM/(2*gm2.FP.probes.position.y[0])*45.
                         ]
                    mp_mean = []
                    for mp_n in range(3):
                        mp_mean.append(mp[mp_n].mean())
                        mp[mp_n] = mp[mp_n] - mp_mean[mp_n]
                    s_t = fp_time[skip:, s_station][:,0] > 0
                    line_dp = plot_ts(fp_time[skip:, s_station][:,0][s_t], mp[0][skip:][s_t], '.', markersize=2, alpha=alpha, label="dipole %.1f ppm" % (mp_mean[0]))
                else:
                    freq_mean = 0
                    freq_n    = 0
                    for radId in radIds:
                        for layer in layers:
                            s = self.selectProbe(radId, layer)&s_station
                            #print("Number", np.argwhere(s).shape)
                            probe = gm2.np.argwhere(s)
                            if len(probe) == 1:
                                if mode == 0:
                                    mean = fp_freq[skip:, s].mean()
                                probe = probe[0]
                                label_ = layer
                                if radId in ["I"]:
                                    label_ += radId+"  "
                                else:
                                    label_ += radId
                                #if mode == 0:
                                label_ += (r" ${\#%03i}$ (%02i:%02i)"  % (probe, self.fp.id['mux'][probe], self.fp.id['round'][probe]))
                                #else:
                                #    label_ += (r" ${\#%03i}$"  % (probe))
                                s_t = fp_time[skip:, s] > 0
                                if len(fp_time[skip:, s][s_t]) > 0:
                                    if mode == 1:
                                        if rel:
                                            mean_ = fp_power[skip:, s][s_t].mean()
                                            plot_ts(fp_time[skip:, s][s_t], 100.*(fp_power[skip:, s][s_t]-mean_)/mean_, '.', markersize=2, label=label_, alpha=alpha)
                                        else:
                                            plot_ts(fp_time[skip:, s][s_t], fp_power[skip:, s][s_t], '.', markersize=2, label=label_, alpha=alpha)
                                        freq_n += 1
                                    else:
                                        if freq_n == 0:
                                            freq_mean = fp_freq[:, s]-mean
                                        else:
                                            freq_mean += fp_freq[:, s]-mean
                                        freq_n += 1
                                        plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-mean, '.', markersize=2, label=label_, alpha=alpha)
                
                #print self.phi[s_station], self.phi[s_station][0], ((phi_+360)%360), yoke, aziId, str(aziId)
                gm2.plt.title("Yoke "+(yoke)+", azi "+str(aziId)+(" at %.0f deg" % ((phi_+360)%360))  )
                if mode == 2:
                    gm2.plt.ylabel("relative dipole [ppm]")
                    gm2.plt.ylim([-3.0,3.0])
                if mode == 1:
                    if rel:
                         gm2.plt.ylabel(r'relative power [%]')
                         if ylim > 0:
                             gm2.plt.ylim([-ylim, ylim])
                    else:
                         gm2.plt.ylabel(r'power')
                if mode == 0:
                    gm2.plt.ylabel(r'$f^{\#} - f_{0}^{\#}$ [Hz]')
                    if (ylim > 0):
                        gm2.plt.ylim([-ylim, ylim])

                if mode == 1:
                     ax2 = gm2.plt.subplot(312, sharex=ax1)
                else:
                     ax2 = gm2.plt.subplot(212, sharex=ax1)
                if mode == 2:
                    s_t = fp_time[skip:, s_station][:,0] > 0
                    line_nq = plot_ts(fp_time[skip:, s_station][:,0][s_t], mp[1][skip:][s_t], '.', markersize=2, label="norm quad %.1f ppm/Hz" % mp_mean[1] , alpha=alpha, color=gm2.sns.color_palette()[1])
                    line_sq = plot_ts(fp_time[skip:, s_station][:,0][s_t], mp[2][skip:][s_t], '.', markersize=2, label="skew quad %1.f ppm/Hz" % mp_mean[2], alpha=alpha, color=gm2.sns.color_palette()[2])
                else:     
                    if freq_n > 0:
                        freq_mean /= freq_n
                        for radId in radIds:
                            for layer in layers:
                                s = self.selectProbe(radId, layer)&s_station
                                probe = gm2.np.argwhere(s)
                                if len(probe) == 1:
                                    if mode == 0:
                                        mean = fp_freq[skip:, s].mean()
                                    s_t = fp_time[skip:, s] > 0
                                    if len(fp_time[skip:, s][s_t]) > 0:
                                        #print "DEBUG"
                                        if mode == 1:
                                            if rel:
                                                 mean_ = fp_length[skip:, s][s_t].mean()
                                                 plot_ts(fp_time[skip:, s][s_t], 100.*(fp_length[skip:, s][s_t]-mean_)/mean_, '.', markersize=2, label=label_, alpha=alpha)
                                            else:
                                                 plot_ts(fp_time[skip:, s][s_t], fp_length[skip:, s][s_t], '.', markersize=2, label=label_, alpha=alpha)
                                        else:
                                            plot_ts(fp_time[skip:, s][s_t], fp_freq[skip:, s][s_t]-mean-freq_mean[skip:][s_t], '.', markersize=2, label=label_, alpha=alpha)
                
                #gm2.plt.xlabel("time")
                if mode == 2:
                    gm2.plt.ylabel('relative quadrupole [ppm/45mm]')
                    gm2.plt.ylim([-1.0,1.0])
                if mode == 1:
                   if rel: 
                       gm2.plt.ylabel('relative length [%]')  
                       if ylim2 > 0:
                           gm2.plt.ylim([-ylim2, ylim2])
                   else:
                       gm2.plt.ylabel('length [s]')    
                if mode == 0:
                   gm2.plt.ylabel(r'$(f^{\#} - f_{0}^{\#}) - f_{\rm{mean}}^{\rm{station}}$ [Hz]')
                   if ylim2 > 0:
                       gm2.plt.ylim([-ylim2, ylim2])

                if mode == 1:
                     ax3 = gm2.plt.subplot(313, sharex=ax1)
                     if freq_n > 0:
                        for radId in radIds:
                            for layer in layers:
                                s = self.selectProbe(radId, layer)&s_station
                                probe = gm2.np.argwhere(s)
                                if len(probe) == 1:
                                    s_t = fp_time[skip:, s] > 0
                                    if len(fp_time[skip:, s][s_t]) > 0:
                                        #if rel:
                                        #    mean_ = fp_chi2[skip:, s][s_t].mean()
                                        #    plot_ts(fp_time[skip:, s][s_t], fp_chi2[skip:, s][s_t], '.', markersize=2, label=label_, alpha=alpha)
                                        #else:
                                        plot_ts(fp_time[skip:, s][s_t], fp_chi2[skip:, s][s_t], '.', markersize=2, label=label_, alpha=alpha)
                     
                     #if rel:
                     #    gm2.plt.ylabel('relative chi2') 
                     #else:
                     gm2.plt.ylabel('chi2')
                     gm2.plt.yscale('log')
                     if ylim3 > 0:
                         gm2.plt.ylim([1e-5, ylim3])
                     
                     #gm2.plt.subplot(312)
                     gm2.plt.setp(ax2.get_xticklabels(), visible=False)
                #gm2.plt.gca().xaxis.t_major_formatter(formatter)
                #ax.xaxis.set_tick_params(rotation=30, labelsize=10)
                gm2.despine()
                if mode == 1:
                    gm2.plt.subplot(311)
                else:
                    gm2.plt.subplot(211)
                gm2.plt.setp(ax1.get_xticklabels(), visible=False)
                #plt.gca().set_xticklabels([]);
                if mode == 2:
                   handles, labels = ax1.get_legend_handles_labels()
                   handles2, labels2 = ax2.get_legend_handles_labels()
                   lgnd = gm2.plt.legend([handles[0], handles2[0], handles2[1]],[labels[0], labels2[0], labels2[1]], loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
                else:
                   lgnd = gm2.plt.legend(loc='upper center', bbox_to_anchor=(0.2, -.22, 0.6, 0.2), ncol=3, fontsize=12)
                for lgndHandl in lgnd.legendHandles:
                    lgndHandl._legmarker.set_markersize(12)
                f.savefig(folder+"/"+dirname+"/"+prefix+"_"+yoke+"_"+str(aziId)+".png")
                if show:
                    gm2.plt.show()
                gm2.plt.close('all')
                #test = 1

        # combine plots to gif
        import os
        if gif:
            os.system("convert -delay 100 "+folder+"/"+dirname+"/"+prefix+"_*_*.png -loop 0 "+folder+"/"+prefix+"_"+dirname+".gif")
            print(dirname+"/"+prefix+"_"+dirname+".gif")

    def temperatures(self, folder="plots", show=False):
        """Generates temperature overview of the period corresponding to the loaded files.
        
        Args:
            show (bool, optional) : if True the plot is shown. Defaults to False.
            folder (str, optional) : folder where the plot is stored. Defaults to 'plots'.

        Returns:
            Stores the plot in 'folder' in the format tmp_STARTRUNtoENDRUN.png.
        
        """
        t = gm2.Temperature(db=True)
        probe = 100
        s = self.fp.time[:,probe] > 0
        start_ts = gm2.np.nanmin(self.fp.time[s,probe]) 
        end_ts   = gm2.np.nanmax(self.fp.time[s,probe])
        f = t.plotAll(start=gm2.util.ts2datetime(gm2.np.array([start_ts]))[0], 
                      end=gm2.util.ts2datetime(gm2.np.array([end_ts]))[0])
        dirname = str(self.runs[0])+"to"+str(self.runs[-1])
        f.savefig(folder+"/tmp_"+dirname+".png")
        print(folder+"/tmp_"+dirname+".png")
        if show:
            gm2.plt.show()
        gm2.plt.close('all')
