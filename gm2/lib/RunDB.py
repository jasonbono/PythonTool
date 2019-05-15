import json
import pandas as pd

class RunDB:
    def __init__(self, fname):
        self.fname = fname
        with open(fname) as f:
            self.rawData = json.load(f)
            self.data = pd.DataFrame(data=self.rawData['body'], columns=self.rawData['header'])
            
            self.data['Run']        = self.data['Run'].apply(int)
            self.data['Start time'] = self.data['Start time'].apply(pd.Timestamp)
            self.data['Stop time']  = self.data['Start time'].apply(pd.Timestamp)
            self.data['Field']      = self.data['Field (kHz)'].apply(float)
            self.data['PSFB']      = self.data['PSFB Curr. (mA)'].apply(float)
            self.data['Trolley pos']      = self.data['Trolley pos'].apply(float)
