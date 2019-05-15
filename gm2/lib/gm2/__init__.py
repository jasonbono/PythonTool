""" This module provides access to the gm2 field data and provides algorithms to handle and manipulate this data.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    FixedProbe
    Trolley
    Interpolation
    util
    plotutil
    constants
    Galil
    Fluxgate
    SurfaceCoil
    Feedback
    FixedProbeWf
    Issues
    Temperature
    E821
    Fourier
    Spikes
    Barcode
    fpViewer
    Runs
    FieldMap
"""


# generall constants, avaiable for all modules
#import gm2.constants as constants
from gm2.constants import * # needed? its already present in util
from gm2.rootbase   import rootbase
from gm2.DB import DB
from gm2.FixedProbe import FixedProbe
from gm2.Galil import Galil
from gm2.Fluxgate import Fluxgate
from gm2.Trolley import Trolley # lower case is already taken by trolley.h
from gm2.SurfaceCoil import SurfaceCoil
from gm2.Feedback import Feedback
from gm2.FixedProbeWf import FixedProbeWf
from gm2.PlungingProbeWf import PlungingProbeWf
from gm2.Issues import Issues
from gm2.Temperature import Temperature
import gm2.plotutil
from gm2.E821 import E821

## Tools
from gm2.Fourier import Fourier
from gm2.Spikes import Spikes
try:
    from gm2.Runs import Runs
    from gm2.OnlineDB import OnlineDB
except ImportError:
    "In order to use the Run class you need to install psycopg2. (pip install psycopg2)."
from gm2.fpViewer import fpViewer
from gm2.FieldMap import FieldMap

## Interpolation Tools
from gm2.Interpolation import Interpolation
from gm2.Barcode import Barcode
from gm2.Tier2Quality import Tier2Quality

# generall stuff outside of modules
try:
    from gm2.plotsettings import *
except:
    pass

import gm2.util

R = gm2.R            #: Magic Radius
PPB2HZ = gm2.PPB2HZ  #: Conversion PPB to HZ
HZ2PPB = gm2.HZ2PPB  #: Conversion HZ to PPB
PPM2HZ = gm2.PPM2HZ  #: Conversion PPM to HZ
HZ2PPM = gm2.HZ2PPM  #: Conversion HZ to PPM   

