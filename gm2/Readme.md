## Jump Start 
Quick usage on newg2@g2field-server

- source ~/scorrodi/gm2setup.sh

## Installation ##
### Prerequisites ###
- python :-), tested with 2.7 and 3 (unfortunetly I am having problems with the version on /cvmfs/gm2.opensciencegrid.org/prod8/.. )
- pyROOT
- scipy (on personal accounts on g2field-server: pip install --user scipy)
    - contains: numpy (on personal accounts on g2field-server: pip install --user numpy)
    - contains: matplotlib (if youn work on g2field-server and you don't have matplitlib, install it locally: pip install --user matplotlib)
- optional : seaborn 
- optional : psycopg2 (for Run class)

### Download ###
- git clone https://corrodis@bitbucket.org/corrodis/gm2.git

### Environment Variables ###
- ARTTFSDIR=PATH-TO-TIER1-DATA (example: /data1/newg2/DataProduction/Nearline/ArtTFSDir/)
- source gm2/thisgm2.sh

### Create and Compile Libraries ###

- cd gm2/lib
- run compile.sh
  - you are prompted to enter a run number which will be used to generate the root classes, the specified run-file needs to be available in ARTTFSDIR and contain the tier1 data format you like to work with (same version)

## Usage ##
### Basics ###
#### Trolley ####
```python
import gm2
tr = gm2.Trolley([3996, 3998]) # TChain is intrnaly used
tr_time, tr_pos, tr_freq = tr. getBasics() # return numpy arrays with size [n_events, n_probes]
```

#### FixedProbes

```python
import gm2
fp = gm2.FixedProbe([3997])
fp_time, fp_freq = fp.getBasics() # return numpy array with size [n_events, n_probes]
sel = fp.id['yoke'] == ord("A")
```

### More Advanced

```python
import gm2
tr = gm2.Trolley([3996, 3998])
def callback():
    return [tr.getTimeGPS(), tr.getTimeNMR(), tr.getTimeBarcode(), tr.getTimeFrequency(3)] # tr.getTimeFrequency(3) uses freq method 3 
tr_t_gps, tr_t_nmr, tr_t_bc, tr_freq = tr.loop(callback)
```

### Under the Hood
```python
import gm2
tr = gm2.Trolley([3996, 3998])
tr.getEntries()                  # root: GetENtries()
tr.load(100)                     # root: GetEntry(100)
tr.getFrequencies()              # returns numpy array of this event with frequencies, dim [n_probes = 17, n_freq_methodes = 6]
tr.data.ProbeFrequency_Frequency # direct access of underling struct
```





