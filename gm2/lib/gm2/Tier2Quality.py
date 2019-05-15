import json
import numpy as np

class Tier2Quality:
    '''Class to read tier2 quality json files'''

    def __init__(self, fname):
        ''' 
        
        Args:
            fname (str) : full path to quality json file.

        Attributes:
            data : the parsed json data.
            datasets (list[str]): list of names of the datasets in this json file. 
        '''
        self.fname = fname
        with open(fname) as json_file:
            self.data = json.load(json_file)

        self.datasets = self.getDatasets()

    def getDatasets(self):
        return self.data.keys()

    def getMagnetPeriods(self, dataset):
        periods = []
        for d in self.data[dataset]:
            periods.append({
                'id' : d['Magnet Period Id'],
                'start' : d['Start Time Stamp']*1e9,
                'end' : d['End Time Stamp']*1e9})
        return periods

    def getTrolleyPeriods(self, dataset, magnetPeriodId):
        periods = []
        for d in self.data[dataset][magnetPeriodId]['Trolley Period List']:
            periods.append({
                'id' : d['Trolley Period Id'],
                'start' : d['Start Time Stamp']*1e9,
                'end' : d['End Time Stamp']*1e9})
        return periods

    def getFieldPeriods(self, dataset, magnetPeriodId, trolleyPeriodId):
        periods = []
        for d in self.data[dataset][magnetPeriodId]['Trolley Period List'][trolleyPeriodId]['Field Period List']:
            periods.append({
                'id' : d['Field Period Id'],
                'start' : d['Start Time Stamp']*1e9})
        return periods

    def getFieldSubPeriods(self, dataset, magnetPeriodId, trolleyPeriodId, fielPeriodId):
        periods = []
        for d in  self.data[dataset][magnetPeriodId]['Trolley Period List'][trolleyPeriodId]['Field Period List'][fielPeriodId]['Sub-period List']:
            periods.append({
                'start' : d['Start Time Stamp']*1e9,
                'end' : d['End Time Stamp']*1e9})
        return periods

    def getVeto(self, dataset, magnetPeriodId, trolleyPeriodId, fielPeriodId, fieldSubPeriodId):
         periods = []
         for d in self.data[dataset][magnetPeriodId]['Trolley Period List'][trolleyPeriodId]['Field Period List'][fielPeriodId]['Sub-period List'][fieldSubPeriodId]['Veto Time Stamp List']:
             periods.append(d*1e9)
         return periods

    def getAllTrolleyPeriods(self):
        '''Get all Trolley Periods from loaded Quality file.
        
        Returns:
            periods (numpy.ndarray([n_periodsx2])) : start and stop timestamps (ns) of trolley periods.
            periods_name (list(str)) : name of trolley periods in the form dataset-magnetCycleId-trolleyCycleId'''
        periods = []
        periods_name = []
        for dataset in self.getDatasets():
            for magnet_n, magnet in enumerate(self.getMagnetPeriods(dataset)):
                for trolley_n, trolley in enumerate(self.getTrolleyPeriods(dataset, magnet_n)):
                    periods.append([trolley['start'], trolley['end']])
                    periods_name.append(dataset + '-' + str(magnet['id']) +'-'+str(trolley['id']))
        return np.array(periods), periods_name

    def getAllFieldSubPeriods(self):
        '''Returns all field sub run periods.
        
        Returns:
            periods (numpy.darray([n_subperids x 2])) : start and stop timestamps (ns) of field sub-periods.
            periods_name (list(str)) : name of field sub-periods.
        '''
        periods = []
        periods_name = []
        for dataset in self.getDatasets():
            for magnet_n, magnet in enumerate(self.getMagnetPeriods(dataset)):
                for trolley_n, trolley in enumerate(self.getTrolleyPeriods(dataset, magnet_n)):
                    for field_n, field in enumerate(self.getFieldPeriods(dataset, magnet_n, trolley_n)):
                        for subField_n, subField in enumerate(self.getFieldSubPeriods(dataset, magnet_n, trolley_n, field_n)):
                            periods.append([subField['start'], subField['end']])
                            periods_name.append(str(field['id'])+".%i" % subField_n)
        return np.array(periods), periods_name

    def getAllFieldSubPeriodsSubset(self):
        """Returns all field sub run periods grouped into field trolley periods and periods""" 
        periods = []
        periods_name = []
        for dataset in self.getDatasets():
            for magnet_n, magnet in enumerate(self.getMagnetPeriods(dataset)):
                for trolley_n, trolley in enumerate(self.getTrolleyPeriods(dataset, magnet_n)):
                    for field_n, field in enumerate(self.getFieldPeriods(dataset, magnet_n, trolley_n)):
                        periods.append([])
                        periods_name.append([])
                        for subField_n, subField in enumerate(self.getFieldSubPeriods(dataset, magnet_n, trolley_n, field_n)):
                            periods[-1].append([subField['start'], subField['end']])
                            periods_name[-1].append(str(field['id'])+".%i" % subField_n)
        return np.array(periods), periods_name


    def getAllVeto(self):
        '''Returns all veto times
        
        Returns:
            times (nump.ndaray([n_veto]))) : list of all veto times (ns).
        '''
        vetos = []
        for dataset in self.getDatasets():
            for magnet_n, magnet in enumerate(self.getMagnetPeriods(dataset)):
                for trolley_n, trolley in enumerate(self.getTrolleyPeriods(dataset, magnet_n)):
                    for field_n, field in enumerate(self.getFieldPeriods(dataset, magnet_n, trolley_n)):
                        for subField_n, subField in enumerate(self.getFieldSubPeriods(dataset, magnet_n, trolley_n, field_n)):
                            for veto in self.getVeto(dataset, magnet_n, trolley_n, field_n, subField_n):
                                vetos.append(veto)
        return np.array(vetos)

    def plot(self):
        import gm2
        from matplotlib.patches import Rectangle
        import matplotlib.dates as mdates
        rect = []
        ylim = [mdates.date2num(gm2.util.ts2datetime(gm2.np.array([self.getAllTrolleyPeriods()[0][-1][0]]))[0]),
                mdates.date2num(gm2.util.ts2datetime(gm2.np.array([self.getAllTrolleyPeriods()[0][0][0]]))[0]),]
        ax = gm2.plt.subplot(111)
        trolley_w = []
        trolley_w_sum = 0
        for i, p in enumerate(self.getAllTrolleyPeriods()[0]):
            #print i
            start = mdates.date2num(gm2.util.ts2datetime(gm2.np.array([p[0]]))[0])
            end = mdates.date2num(  gm2.util.ts2datetime(gm2.np.array([p[1]]))[0])
            if start < ylim[0]:
                ylim[0] = start
            if end > ylim[1]:
                ylim[1] = end
            width = end - start
            trolley_w.append(p[1]-p[0])
            trolley_w_sum += p[1]-p[0]

            # Plot rectangle
            rect.append(Rectangle((start, 0.1), width, 0.9, color=gm2.sns.color_palette()[i], alpha=0.3))
            ax.add_patch(rect[-1])   


        trolley_n = -1
        for i, p in enumerate(self.getAllFieldSubPeriodsSubset()[0]):
            if int(float(self.getAllFieldSubPeriodsSubset()[1][i][0])) == 0:
                trolley_n += 1
            start = p[0][0]
            end   = p[-1][1]
            width = end -  start
            #print self.getAllFieldSubPeriodsSubset()[1][i][0]
            # Plot rectangle
            start_ = mdates.date2num(gm2.util.ts2datetime(gm2.np.array([start]))[0])
            end_   = mdates.date2num(  gm2.util.ts2datetime(gm2.np.array([end]))[0]) 
            width_ = end_ - start_

            rect.append(Rectangle((start_, -1), width_, 0.9, color=gm2.sns.color_palette()[i%10], alpha=0.3))
            #print width
            ax.add_patch(rect[-1])   

            integral = 0
            for pp in p:
                 integral += pp[1] - pp[0]
            print(self.getAllFieldSubPeriodsSubset()[1][i][0], "%.1f" % (100*width/trolley_w_sum), "%.1f" % (100.*width/trolley_w[trolley_n]),  "%.1f" % ((1.0-integral/(end-start))*100))


        # assign date locator / formatter to the x-axis to get proper labels
        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # set the limits
        gm2.plt.xlim(ylim)
        gm2.plt.ylim([-1.7, 1.2])

        # go
        gm2.despine()
        gm2.plt.show()
