import Trolley
import util

t = Trolley.Trolley([3998])

def callback():
    return t.getFrequencys(True), t.getTimeGPSs(), t.getMultipoles(), t.getPhis()
freq, time, multipole, phi = t.loop(callback)


