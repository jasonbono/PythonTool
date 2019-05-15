#%load_ext autoreload
#%autoreload 2

import matplotlib.pyplot as plt
import numpy as np

#import FixedProbe
#t = FixedProbe.FixedProbe([3721])
#t.getEntry(0)
#t.show(6)

#import sys
#if 'Trolley' not in sys.modules:

import os
if not os.path.isfile('driftInTrolleyRun.dat'):

	import Trolley
	t0 = Trolley.Trolley([5216])
	t0.getEntry(42)
	t0.show(6)

	#import os.path
	#if not os.path.isfile('tmp.data'):

	times0, freq0, freqUnc0, phi0, phiUnc0 = t0.getFrequencyHistory()

	t1 = Trolley.Trolley([5217])
	times1, freq1, freqUnc1, phi1, phiUnc1 = t1.getFrequencyHistory()


	t2 = Trolley.Trolley([5218])
	times2, freq2, freqUnc2, phi2, phiUnc2 = t2.getFrequencyHistory()

	with open('driftInTrolleyRun.dat', 'wb') as fp:
	    np.savez(fp, times0=times0, freq0=freq0, freqUnc0=freqUnc0, phi0=phi0, phiUnc0=phiUnc0,
			 times1=times1, freq1=freq1, freqUnc1=freqUnc1, phi1=phi1, phiUnc1=phiUnc1,
			 times2=times2, freq2=freq2, freqUnc2=freqUnc2, phi2=phi2, phiUnc2=phiUnc2)

else:
	with open('driftInTrolleyRun.dat', 'rb') as fp:
	    ff = np.load(fp) 
            locals().update(ff)



from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('driftInTrolleyRun.pdf') as pdf:
    for p in range(1):
	print "Processing Probe", p
	#p = 0 # probe



	import seaborn as sns
	sns.set_style("ticks")
	sns.set_context("notebook", font_scale=1.4)
	fig_size = np.array([8, 5.5])


	f = plt.figure(figsize=fig_size*2.0)
	ax1 = plt.subplot2grid((3,4), (0, 0), colspan=3)
	ax2 = plt.subplot2grid((3,4), (1, 0), colspan=3)
	ax3 = plt.subplot2grid((3,4), (2, 0), colspan=3)
	ax4 = plt.subplot2grid((3,4), (1,3))
	ax5 = plt.subplot2grid((3,4), (2,3))
	#ax6 = plt.subplot2grid((3,4), (0,3))
	#ax1 = plt.gca()

	ax1.errorbar(phi0[:, p, 0], freq0[:, p, 0]/1e3, xerr=phiUnc0[:, p, 0], yerr=freqUnc0[:, p, 0]/1e3, fmt='-', label="CW: garage to drive")
	ax1.errorbar(phi2[:, p, 0], freq2[:, p, 0]/1e3, xerr=phiUnc2[:, p, 0], yerr=freqUnc2[:, p, 0]/1e3, fmt='-', label="CW: drive to garage")
	ax1.errorbar(phi1[:, p, 0], freq1[:, p, 0]/1e3, xerr=phiUnc1[:, p, 0], yerr=freqUnc1[:, p, 0]/1e3, fmt='-', label="CCW: drive to drive")
	#f.subplots_adjust(hspace=0)
	#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


	from scipy.interpolate import interp1d
	ccw        = interp1d(phi1[:, p, 0], freq1[:, p, 0])
	ccwPhiUnc  = interp1d(phi1[:, p, 0], phiUnc1[:, p, 0])
	ccwFreqUnc = interp1d(phi1[:, p, 0], freqUnc1[:, p, 0])
	ccwTimes    = interp1d(phi1[:, p, 0], times1[:,p])

	N=100
	weight = np.concatenate((np.arange(N),np.arange(N,0,-1)))/(1.0 * N**2)


	s = (phi0[:,p,0]>phi1[:,p,0].min())&(phi0[:,p,0]<phi1[:,p,0].max())&(times0[:,p]>10)
	xx = phi0[:,p,0][s]
	ax2.errorbar(xx, freq0[s,p,0]-ccw(xx), xerr=np.sqrt(phiUnc0[s,p,0]**2 + ccwPhiUnc(xx)**2), yerr=np.sqrt(phiUnc0[s,p,0]**2 + ccwFreqUnc(xx)**2), fmt=' ', label="CW-CCW: garage to drive" )
	ax2.plot(np.convolve(xx, weight, mode='valid'), np.convolve(freq0[s,p,0]-ccw(xx), weight,  mode='valid'), '--', color='gray')

	ax4.hist(freq0[s,p,0]-ccw(xx), bins=np.arange(-300,300,10), orientation='horizontal', histtype='step') 

	s = (phi2[:,p,0]>phi1[:,p,0].min())&(phi2[:,p,0]<phi1[:,p,0].max())&(times2[:,p]>10)
	xx = phi2[:,p,0][s]
	ax2.errorbar(xx, freq2[s,p,0]-ccw(xx), xerr=np.sqrt(phiUnc2[s,p,0]**2 + ccwPhiUnc(xx)**2), yerr=np.sqrt(phiUnc2[s,p,0]**2 + ccwFreqUnc(xx)**2), fmt=' ', label="CW-CCW: drive to garage" )
	ax2.plot(np.convolve(xx, weight, mode='valid'), np.convolve(freq2[s,p,0]-ccw(xx), weight, mode='valid'), '--', color='gray')

	ax4.hist(freq2[s,p,0]-ccw(xx), bins=np.arange(-300,300,10), orientation='horizontal', histtype='step') 

	ax3.plot([-40, 110],[0,0], '--', color='gray', linewidth=sns.plotting_context()['lines.linewidth']*0.5)
	# Time
	s = (phi0[:,p,0]>phi1[:,p,0].min())&(phi0[:,p,0]<phi1[:,p,0].max())
	xx = phi0[:,p,0][s]
	dt = (times0[s,0] - ccwTimes(xx))/1e9/60
	ax3.errorbar(dt, freq0[s,p,0]-ccw(xx), yerr=np.sqrt(phiUnc0[s,p,0]**2 + ccwFreqUnc(xx)**2), fmt=' ', label="CW-CCW: garage to drive" )

	#ax3.plot(np.convolve((times0[s,0] - ccwTimes(xx))/1e9/60, weight, mode='valid'), np.convolve(freq0[s,p,0]-ccw(xx), weight, mode='valid'), '--', color='gray')

	ss = (dt > -30)&(dt < -10)
	ax5.hist((freq0[s,p,0]-ccw(xx))[ss], bins=np.arange(-300,300,10), orientation='horizontal', histtype='step', label=r'-30<$\Delta$t<-10') 

	s = (phi2[:,p,0]>phi1[:,p,0].min())&(phi2[:,p,0]<phi1[:,p,0].max())
	xx = phi2[:,p,0][s]
	dt = (times2[s,0] - ccwTimes(xx))/1e9/60
	ax3.errorbar(dt, freq2[s,p,0]-ccw(xx), xerr=np.sqrt(phiUnc2[s,p,0]**2 + ccwPhiUnc(xx)**2), yerr=np.sqrt(phiUnc2[s,p,0]**2 + ccwFreqUnc(xx)**2), fmt=' ', label="CW-CCW: drive to garage" )

	#ax3.plot(np.convolve((times2[s,0] - ccwTimes(xx))/1e9/60, weight, mode='valid'), np.convolve(freq2[s,p,0]-ccw(xx), weight, mode='valid'), '--', color='gray')

	ss = (dt > 80)&(dt < 100)
	ax5.hist((freq2[s,p,0]-ccw(xx))[ss], bins=np.arange(-300,300,10), orientation='horizontal', histtype='step', label=r'80<$\Delta$t<100') 

	ax1.legend()
	#ax1.set_xlabel("phi [rad]")

	ax1.set_xlabel("phi [rad]")
	ax1.xaxis.set_label_coords(0.5, -0.03)
	ax2.set_xlabel("phi [rad]")
	ax2.xaxis.set_label_coords(0.5, -0.03)
	ax2.set_ylabel("freq difference [Hz]")
	ax2_2 = ax2.twinx()
	ax2_2.set_ylabel("[ppm]", rotation=90)
	ax2_2.yaxis.set_label_coords(1.02, 1.14)
	ax2_2.set_ylim([-300./52000.0*1000,300/52000.0*1000])
	ax1.set_ylabel("freq [kHz]")
	ax1.set_ylim([48, 56])
	ax2.set_ylim([-300, 300])
	ax4.set_ylim([-300,300])
	#ax4.axes.get_xaxis().set_visible(False)
	#ax4.axes.get_yaxis().set_visible(False)
	#ax4.axes.get_xaxis().set_ticks([])
	#ax4.axes.get_yaxis().set_ticks([])
	ax4.xaxis.set_ticklabels([])
	ax4.yaxis.set_ticklabels([])
	ax5.set_xlabel("counts")
	ax5.set_ylim([-300,300])
	#ax4.axes.get_xaxis().set_visible(False)
	#ax4.axes.get_yaxis().set_visible(False)
	#ax4.axes.get_xaxis().set_ticks([])
	#ax4.axes.get_yaxis().set_ticks([])
	ax5.xaxis.set_ticklabels([])
	ax5.yaxis.set_ticklabels([])
	ax5.set_xlabel("counts")
	ax5.legend()
	ax3.set_xlabel(r'time difference $\Delta$t [min]')
	ax3.set_ylabel("freq difference [Hz]")
	ax3.set_xlim([-40,110])
	ax3.set_ylim([-300, 300])
	ax2.set_xlim([-2,5])
	ax1.set_xlim([-2,5])

	ax1.text(1.1, 0.9, "Trolley Run\n\nProbe: "+str(p), fontsize=18,
        verticalalignment='top', transform=ax1.transAxes)


        #plt.show()
   
	plt.show()
	sns.despine(f)
	pdf.savefig(f)
        #plt.cgf()
