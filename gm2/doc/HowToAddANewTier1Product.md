## How to Add a New Tier1 Product
### Step 1: Generation of Libraries/Dictionaries 
a )  given the new product has the following path in root (e.g. TBrowser) directory: TreeGenFluxgate, branch: fluxgate modify "lib/generateClass.C"
```
TChain * chainFluxgate = new TChain("fluxgate", "");
chainFluxgate->Add((path+"/"+fname+"/TreeGenFluxgate/fluxgate").c_str());
chainFluxgate>MakeClass("fluxgate");
```
This will generate fluxgate.C and fluxgate.h

b) Unfortunately I had troubles with the root data types, therefore I manually translate to c++ types. This is done in "lib/compile.sh". Add the new product to the list of products:
```
products=("fixedProbe" "trolley" "galil" "fluxgate")
```
Now, regenerate the libraries by running compile.sh (this invokes generateClass.C).

### Step 2: Create Class
Create new Class in "lib/gm2/Fluxgate.py" which inherits from rootbase. 
##### a) Minimal Working Example
```python
from gm2 import rootbase

class Fluxgate(rootbase, object):
    def __init__(self, runs = []):
        self.runs = runs
        self.loadSettings() # load local settings
        # name of library to load 'fluxgate'
        super(Fluxgate, self).__init__('fluxgate', None) 
        self.loadFiles() # opens the root files,
        
    def loadSettings(self):                                   
        """ Fluxgate specific settings """              
        # only required setting is the path in the root file
        self.fname_path     = "TreeGenFluxgate/fluxgate"
        # and potential more product specific settings... 
```

What happens under the hook:
```python
from ROOT import gInterpreter, gSystem, TChain 
gInterpreter.ProcessLine('#include "'+path+'/lib/fluxgate.hh"') 
gSystem.Load(path+'/lib/libfluxgate.[so or dylib];')
self.data = ROOT.fluxgate(tchain)
```
Where tchain is ROOT.TChain('fluxgate') with all the specified files loaded.

##### b) Add the Class to the gm2 Library
To add the new class to the library you have to add it ro "lib/gm2/\_\_init\_\_.py"
```python
from gm2.Fluxgate import Fluxgate 
```

No you should be able to access the data in a root-cint wise style.
```python
In [1]: import gm2
In [2]: fg = gm2.Fluxgate([3997])
In [3]: fg.getEntries()
Out[3]: 293
In [4]: fg.load(29)
In [5]: fg.data.waveform_eff_rate
Out[5]: 500.0
```

##### c) Add numpy access functions
A few examples:
```python
def getTimeGPS(self):    
    return self.data.waveform_gps_clock

def getTheta(self):
    return np.frombuffer(self.data.waveform_fg_theta, dtype='double')

def getWaveform(self):
    return np.frombuffer(self.data.waveform_trace, dtype='double').reshape([24, self.tr_l])
```

This can now be used in the following way:
```python
fg = gm2.Fluxgate([runno])
def callback():
    return [fg.getTimeGPS(), fg.getWaveform()]
fg_time, fg_wfs =  fg.loop(callback)
```
