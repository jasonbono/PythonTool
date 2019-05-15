import gm2
from gm2 import np

pos_fp = (gm2.FP.probes.position.r, gm2.FP.probes.position.theta)
pos_tr = (gm2.TR.probes.position.r, gm2.TR.probes.position.theta)

trb = np.array(
[[   0.076923,   -0.      ,    0.      ,    0.      ,    0.      ,    0.076923,
     0.076923,    0.076923,    0.076923,    0.076923,    0.076923,    0.076923,
     0.076923,    0.076923,    0.076923,    0.076923,    0.076923],
 [   0.      ,    0.      ,   -0.      ,   -0.      ,    0.      ,   -0.,
     0.107143,    0.185577,    0.214286,    0.185577,    0.107143,    0.,
    -0.107143,   -0.185577,   -0.214286,   -0.185577,   -0.107143],
 [  -0.      ,    0.      ,   -0.      ,   -0.      ,    0.      ,   -0.214286,
    -0.185577,   -0.107143,   -0.      ,    0.107143,    0.185577,    0.214286,
     0.185577,    0.107143,   -0.      ,   -0.107143,   -0.185577],
 [   0.      ,   -0.      ,    0.      ,   -0.      ,   -0.      ,    0.,
    -0.238599,   -0.238599,    0.      ,    0.238599,    0.238599,    0.,
    -0.238599,   -0.238599,    0.      ,    0.238599,    0.238599],
 [  -0.      ,   -0.066117,    0.066117,   -0.066117,    0.066117,   -0.264146,
    -0.13259 ,    0.13259 ,    0.264146,    0.13259 ,   -0.13259 ,   -0.264146,
    -0.13259 ,    0.13259 ,    0.264146,    0.13259 ,   -0.13259 ],
 [   0.      ,   -0.      ,    0.      ,   -0.      ,   -0.      ,    0.,
    -0.354227,   -0.      ,    0.354227,   -0.      ,   -0.354227,    0.,
     0.354227,    0.      ,   -0.354227,    0.      ,    0.354227],
 [   0.      ,    0.      ,    0.      ,   -0.      ,   -0.      ,    0.354227,
    -0.      ,   -0.354227,   -0.      ,    0.354227,    0.      ,   -0.354227,
     0.      ,    0.354227,    0.      ,   -0.354227,   -0.      ],
 [  -3.587428,   11.659142,   11.659142,   11.659142,   11.659142,   -3.617791,
    -3.572247,   -3.572247,   -3.617791,   -3.572247,   -3.572247,   -3.617791,
    -3.572247,   -3.572247,   -3.617791,   -3.572247,   -3.572247],
 [  -0.      ,   -0.      ,   -0.      ,    0.      ,    0.      ,    0.,
     0.347836,   -0.347836,    0.      ,    0.347836,   -0.347836,   -0.,
     0.347836,   -0.347836,   -0.      ,    0.347836,   -0.347836],
 [  -0.      ,    0.      ,   74.951627,    0.      ,  -74.951627,    0.,
    -3.220578,  -10.649299,  -15.810109,  -10.649299,   -3.220578,    0.,
     3.220578,   10.649299,   15.810109,   10.649299,    3.220578],
 [  -0.      ,  -74.951627,   -0.      ,   74.951627,    0.      ,   15.810109,
    10.649299,    3.220578,    0.      ,   -3.220578,  -10.649299,  -15.810109,
   -10.649299,   -3.220578,   -0.      ,    3.220578,   10.649299],
 [   0.      ,   -0.005646,    0.005646,   -0.005646,    0.005646,   -0.375461,
     0.376872,   -0.376872,    0.375461,   -0.376872,    0.376872,   -0.375461,
     0.376872,   -0.376872,    0.375461,   -0.376872,    0.376872],
 [  -0.      ,   -0.      ,    0.      ,    0.      ,   -0.      ,    0.      ,    0.,
    -0.      ,   -0.      ,   -0.      ,   -0.      ,   -0.      ,   -0.      ,    0.,
     0.      ,    0.      ,    0.      ],
 [   0.      ,   -0.      , -123.899629,   -0.      , 123.899629 ,  -0.,
     5.807795,   16.76566 ,   27.103044,   16.76566 ,   5.807795 ,  -0.,
    -5.807795,  -16.76566 ,  -27.103044,  -16.76566 ,   -5.807795],
 [  -0.      , -123.899629,   -0.      ,  123.899629,    0.      ,   27.103044,
    16.76566 ,    5.807795,    0.      ,   -5.807795,  -16.76566 ,  -27.103044,
   -16.76566 ,   -5.807795,   -0.      ,    5.807795,   16.76566 ],
 [   9.803048,  -31.859905,  -31.859905,  -31.859905,  -31.859905,   11.130544,
     9.1393  ,    9.1393  ,   11.130544,    9.1393  ,    9.1393  ,   11.130544,
     9.1393  ,    9.1393  ,   11.130544,    9.1393  ,    9.1393  ],
 [  -0.      ,   -0.      ,   -0.      ,    0.      ,    0.      ,    0.,
    -0.127291,    0.127291,    0.      ,   -0.127291,    0.127291,   -0.,
    -0.127291,    0.127291,   -0.      ,   -0.127291,    0.127291]])



fpb = np.array([np.array([1, 1, 1, 1, 1, 1])/6.0,
                np.array([1, 0,-1, 1, 0,-1])/-2.67,
                np.array([1, 1, 1,-1,-1,-1])/10.27,
                np.array([1, 0,-1,-1, 0, 1])/-9.13,
                np.array([1,-2, 1, 1,-2, 1])/1.78,
                np.array([1,-2, 1,-1, 2,-1])/9.13])
 
def m1_fp6(f):
    return f.sum()/6.

def m2_fp6(f):
    return (f[0] - f[2] + f[3] - f[5])/60.0/2.0

def m3_fp6(f):
    return (f[0:3].sum() - f[3:6].sum())/77.0/6.0

def m4_fp6(f):
    return (f[0] - f[2] - f[3] + f[5])/9240.0

def m5_fp6(f):
    return (f[0] - 2*f[1] + f[2] + f[3] - 2*f[4] + f[5])/3600.0

def m6_fp6(f):
    return (f[0] - 2*f[1] + f[2] - f[3] + 2*f[4] - f[5])/831600

def m1_fp4(f):
    return (f[1] + f[2] + f[4] + f[5])/4.

def m2_fp4(f):
    return (f[1] - f[2] + f[4] - f[5])/30./2.0

def m3_fp4(f):
    return (f[1:3].sum() - f[4:6].sum())/77.0/4.0

def m4_fp4(f):
    return (f[1] - f[2] - f[4] + f[5])/9240.0

def m1_tr(f):
    return f.sum()/17.0

def m2_tr(f):
    return (f[2]/17.5 + f[6]/17.5 +  f[7]/ 30.310889 + f[8]/35.0  + f[9]/ 30.310889+ f[10]/17.5 - f[4]/17.5 + f[12]/17.5 +  f[13]/ 30.310889 + f[14]/35.0  + f[15]/ 30.310889+ f[16]/17.5)/2.0

def m3_tr(f):
    return (f[1]/17.5 + f[7]/17.5 +  f[6]/ 30.310889 + f[5]/35.0  + f[16]/ 30.310889+ f[15]/17.5 - f[3]/17.5 + f[9]/17.5 +  f[10]/ 30.310889 + f[11]/35.0  + f[12]/ 30.310889+ f[13]/17.5)/2.0

def m4_tr(f):
    return (f[6] + f[7] - f[9] - f[10] + f[12] + f[13] - f[15] - f[16])/8487.0

def m5_tr(f):
    return (f[7] + f[8] + f[9] + f[13] + f[14] + f[15]  -2*f[0] - 2*f[1] - 2*f[3] - 2*f[5] - 2*f[11])/11025.0

def m6_tr(f):
    return (2*f[1] + 2*f[5] - 2*f[3] - 2*f[11] - f[7] + f[9] + f[13] - f[15])/364437.4

def printMp(func, name, pos):
    print(name,  end="\t")
    for i in range(mps.shape[0]):
        print( '%.1e' % func(gm2.util.multipole(pos, *mps[i,:])), end="\t")
    print("")



at = 45.0
r = np.array([1, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5, at**6, at**6, at**7, at**7, at**8, at**8])
mps = np.diag(1./r)

#for m in mps[:,0]/45.:

'''
   |        |        |      fit fp     |      |                  fit try                    |
   | m_fp^6 | m_fp^4 | m=1 | m=3 | m=5 | m_tr | m=1 | m=3 | m=5 | m=7 | m=11 | m=13 | m=15 | m=17 |
m1 |
m2 |
m3 |
m4 |
m5 |
m6 |



'''

latex = True

for i in range(17):
    m = mps[:,i]
    fp_ = gm2.util.multipole(pos_fp, *m)
    tr_ = gm2.util.multipole(pos_tr, *m)
    print("")
    if latex:
        print("\multicolumn{15}{c}{"+str(i)+"}\\ ")
    else:
        print("----", i, "----")

    r = np.array([1, at**1, at**1, at**2, at**2, at**3, at**3, at**4, at**4, at**5, at**5, at**6, at**6, at**7, at**7, at**8, at**8])

    fit_fp = []
    for j in range(1,7,2):
        try:
            fit_fp.append(np.array(gm2.util.getFpMultipole(pos_fp, fp_, n=j)))
        except:
            fit_fp.append(np.full([j], np.nan))
    fit_tr = []
    for j in range(1,18,2):
        fit_tr.append(np.array(gm2.util.getTrMultipole(tr_, n=j)))
    string = ""
    if latex:
        string += "   |        |        | \multicolumn{3}{c}{fit fixed probe} |        | \multicolumn{8}{c}{fit trolley} |"
    else:
        string += "   |        |        |           fit fp         |        |                                     fit try                           |"

    string += "\\\\ \n" if latex else "\n"
    if latex:
        string += "   | $m_{\\rm{fp}}^6$ | $m_{\\rm{fp}}^4$ | $m=1$  |  $m=3$   |  $m=5$   |  $m_{\\rm{try}}$  |  $m=1$   |  $m=3$   |  $m=5$   |  $m=7$   |  $m=11$  |  $m=13$  |  $m=15$  |  $m=17$  |"
    else:
        string += "   | m_fp^6 | m_fp^4 |  m=1   |  m=3   |  m=5   |  m_tr  |  m=1   |  m=3   |  m=5   |  m=7   |  m=11  |  m=13  |  m=15  |  m=17  |"

    string += "\\\\ \n" if latex else "\n"
    if latex:
        string += "\hline \\\\ \n"
    string += "m1 | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[0,:]).sum(), m1_fp4(fp_), 
                                                                                                               fit_fp[0][0], fit_fp[1][0], fit_fp[2][0], 
                                                                                         (tr_*trb[0,:]).sum(), fit_tr[0][0], fit_tr[1][0], fit_tr[2][0], fit_tr[3][0], 
                                                                                                               fit_tr[4][0], fit_tr[5][0], fit_tr[6][0], fit_tr[7][0]    
                                                                                         )

    string += "\\\\ \n" if latex else "\n"
    string += "m2 | %+6.2f | %+6.2f | ------ | %+6.2f | %+6.2f | %+6.2f | ------ | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[1,:]).sum(), m2_fp4(fp_), 
                                                                                                                                           fit_fp[1][1], fit_fp[2][1], 
                                                                                         (tr_*trb[1,:]).sum()           , fit_tr[1][1], fit_tr[2][1], fit_tr[3][1], 
                                                                                                               fit_tr[4][1], fit_tr[5][1], fit_tr[6][1], fit_tr[7][1]    
                                                                                         )
    string += "\\\\ \n" if latex else "\n"
    string += "m3 | %+6.2f | %+6.2f | ------ | %+6.2f | %+6.2f | %+6.2f | ------ | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[2,:]).sum(), m3_fp4(fp_), 
                                                                                                                                           fit_fp[1][2], fit_fp[2][2], 
                                                                                         (tr_*trb[2,:]).sum(),            fit_tr[1][2], fit_tr[2][2], fit_tr[3][2], 
                                                                                                               fit_tr[4][2], fit_tr[5][2], fit_tr[6][2], fit_tr[7][2]    
                                                                                         )
    string += "\\\\ \n" if latex else "\n"
    string += "m4 | %+6.2f | %+6.2f | ------ | ------ | %+6.2f | %+6.2f | ------ | ------ | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[3,:]).sum(), m4_fp4(fp_),          
                                                                                                                                                         fit_fp[2][3], 
                                                                                         (tr_*trb[3,:]).sum(),                       fit_tr[2][3], fit_tr[3][3], 
                                                                                                               fit_tr[4][3], fit_tr[5][3], fit_tr[6][3], fit_tr[7][3]    
                                                                                         )
    string += "\\\\ \n" if latex else "\n"
    string += "m5 | %+6.2f | ------ | ------ | ------ | %+6.2f | %+6.2f | ------ | ------ | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[4,:]).sum(),                       
                                                                                                                                                         fit_fp[2][4], 
                                                                                         (tr_*trb[4,:]).sum(),                       fit_tr[2][4], fit_tr[3][4], 
                                                                                                               fit_tr[4][4], fit_tr[5][4], fit_tr[6][4], fit_tr[7][4]    
                                                                                         )
    string += "\\\\ \n" if latex else "\n"
    string += "m6 | %+6.2f | ------ | ------ | ------ | ------ | %+6.2f | ------ | ------ | ------ | %+6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |" % ((fp_*fpb[5,:]).sum(), 
                                                                                         (tr_*trb[5,:]).sum(),                                     fit_tr[3][5], 
                                                                                                               fit_tr[4][5], fit_tr[5][5], fit_tr[6][5], fit_tr[7][5]    
                                                                                         )
    if latex:
        string = string.replace("|","&")
    print(string)


fpb_ = np.array([gm2.util.multipole(pos_fp, 1,     0,    0),
                 gm2.util.multipole(pos_fp, 0, 1./45,    0),
                 gm2.util.multipole(pos_fp, 0,    0, 1./45),
                 gm2.util.multipole(pos_fp, 0,    0,     0, 1./45**2,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0, 1./45**2),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0, 1./45**3,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0, 1./45**3),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0, 1./45**4,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0, 1./45**4),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0, 1./45**5,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**5),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0, 1./45**6,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0, 1./45**6),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0, 1./45**7,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0, 1./45**7), 
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**8,     0),
                 gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0, 1./45**8)]).T


fpb6_ = np.array([gm2.util.multipole(pos_fp, 1,     0,    0),
                  gm2.util.multipole(pos_fp, 0, 1./45,    0),
                  gm2.util.multipole(pos_fp, 0,    0, 1./45),
                  gm2.util.multipole(pos_fp, 0,    0,     0,     0, 1./45**2),
                  gm2.util.multipole(pos_fp, 0,    0,     0, 1./45**2,     0),
                  gm2.util.multipole(pos_fp, 0,    0,     0,     0,      0,  0, 1./45**3)]).T

trb6_ = np.array([gm2.util.multipole(pos_tr, 1,     0,    0),
                  gm2.util.multipole(pos_tr, 0, 1./45,    0),
                  gm2.util.multipole(pos_tr, 0,    0, 1./45),
                  gm2.util.multipole(pos_tr, 0,    0,     0,     0, 1./45**2),
                  gm2.util.multipole(pos_tr, 0,    0,     0, 1./45**2,     0),
                  gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,   0, 1./45**3)]).T




trb_ = np.array([gm2.util.multipole(pos_tr, 1,     0,    0),
                 gm2.util.multipole(pos_tr, 0, 1./45,    0),
                 gm2.util.multipole(pos_tr, 0,    0, 1./45),
                 gm2.util.multipole(pos_tr, 0,    0,     0, 1./45**2,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0, 1./45**2),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0, 1./45**3,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0, 1./45**3),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0, 1./45**4,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0, 1./45**4),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0, 1./45**5,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**5),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0, 1./45**6,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0, 1./45**6),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0, 1./45**7,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0, 1./45**7), 
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0, 1./45**8,     0),
                 gm2.util.multipole(pos_tr, 0,    0,     0,     0,      0,     0,     0,     0,     0,      0,     0,     0,     0,     0,     0,     0, 1./45**8)]).T





print(np.array_str(trb[[0,1,2,3,4,6],:].dot(trb6_), precision=2, suppress_small=True))


'''
print("m1 %.2f" % m1_fp6(fp_), end=" ")
print("m2 %.2f" % m2_fp6(fp_), end=" ")
print("m3 %.2f" % m3_fp6(fp_), end=" ")
print("m4 %.2f" % m4_fp6(fp_), end=" ")
print("m5 %.2f" % m5_fp6(fp_), end=" ")
print("m6 %.2f" % m6_fp6(fp_), end=" ")
print(" ")
print("m1 %.2f" % m1_fp4(fp_), end=" ")
print("m2 %.2f" % m2_fp4(fp_), end=" ")
print("m3 %.2f" % m3_fp4(fp_))
print("m4 %.2f" % m4_fp4(fp_))
print(" ")
for j in range(1,7,2):
    mm = gm2.util.getFpMultipole(pos_fp, fp_, n=j)
    for i, p in enumerate(mm):
        print("fp fit (n=%i), m=%i : %.2f" % (j, i+1, p/(45.0**((i+1)//2))))
print(" ")



print("Trolley")
print("m1 %.2f" % (tr_*trb[0,:]).sum())
print("m2 %.2f" % (tr_*trb[1,:]).sum())
print("m3 %.2f" % (tr_*trb[2,:]).sum())
print("m4 %.2f" % (tr_*trb[3,:]).sum())
print("m5 %.2f" % (tr_*trb[4,:]).sum())
print("m6 %.2f" % (tr_*trb[5,:]).sum())
'''


'''
printMp(m1_fp6, 'm1_fp6', pos_fp)
printMp(m1_fp4, 'm1_fp4', pos_fp)
printMp(m1_tr,  'm1_tr' , pos_tr)
printMp(m2_fp6, 'm2_fp6', pos_fp)
printMp(m2_fp4, 'm2_fp4', pos_fp)
printMp(m2_tr,  'm2_tr',  pos_tr)
printMp(m3_fp6, 'm3_fp6', pos_fp)
printMp(m3_fp4, 'm3_fp4', pos_fp)
printMp(m3_tr,  'm3_tr',  pos_tr)
printMp(m4_fp6, 'm4_fp6', pos_fp)
printMp(m4_fp4, 'm4_fp4', pos_fp)
printMp(m4_tr,  'm4_tr',  pos_tr)
printMp(m5_fp6, 'm5_fp6', pos_fp)
printMp(m5_tr,  'm5_tr',  pos_tr)
printMp(m6_fp6, 'm6_fp6', pos_fp)
printMp(m6_tr,  'm6_tr',  pos_tr)



'''

xx = np.arange(0,50,0.1)
for i in range(10): 
    plt.plot(xx, (xx/45.)**i, label="n=%i" % i) 

plt.plot([35,35], [0,2]), '--')
plt.legend(ncol=2)
gm2.despine()
plt.show()

