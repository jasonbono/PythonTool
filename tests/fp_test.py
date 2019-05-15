from ROOT import gInterpreter, gSystem, TChain
gInterpreter.ProcessLine('#include "/Users/bono/Desktop/PythonTool/newest/gm2/lib/fixedProbe.hh"')
gSystem.Load('/Users/bono/Desktop/PythonTool/newest/gm2/lib/libfixedProbe')
from ROOT import fixedProbe
chain = TChain("foo")
fp = fixedProbe(chain)
fp.Header_MuxId
