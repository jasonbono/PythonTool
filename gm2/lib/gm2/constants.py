import numpy as np
import ROOT

# general settings
PPM_HZ = 61.79
PPB_HZ = PPM_HZ/1000.
R      = 7112.0      #: Magic Radius
PPB2HZ = PPB_HZ      #: Conversion PPB to HZ
HZ2PPB = 1.0/PPB_HZ  #: Conversion HZ to PPB
PPM2HZ = PPM_HZ      #: Conversion PPM to HZ
HZ2PPM = 1.0/PPM_HZ  #: Conversion HZ to PPM


# multipole
class MP:
    r = 45.0 # mm

# FixedProbes
class FP:
    class probes:
        class position:
            layer = np.array([
                'T', 'T', 'T', 
                'B', 'B', 'B'])
            rad = np.array([
                'I', 'M', 'O',
                'I', 'M', 'O'])
            xy = np.array([
                [-30.0,  77.0],
                [  0.0,  77.0],
                [ 30.0,  77.0],
                [-30.0, -77.0],
                [  0.0, -77.0],
                [ 30.0, -77.0]])
            x     = xy[:,0]
            y     = xy[:,1]
            r     = np.sqrt(   xy[:,1]**2 + xy[:,0]**2) 
            theta = np.arctan2(xy[:,1],     xy[:,0])

            @staticmethod
            def getR(layer_, rad_):
                return FP.probes.position.r[(FP.probes.position.layer==layer_) & (FP.probes.position.rad==rad_)][0]
            @staticmethod
            def getTheta(layer_, rad_):
                return FP.probes.position.theta[(FP.probes.position.layer==layer_) & (FP.probes.position.rad==rad_)][0]
    class stations:
        azimuth = np.array([-0.20943951, -0.13962632, -0.03490655,  0.05235983,  0.13962632,
                             0.20943951,  0.31415928,  0.3839724 ,  0.48869224,  0.57595868,
                             0.66322511,  0.7330383 ,  0.83775803,  0.90757122,  1.01229094,
                             1.09955746,  1.18682393,  1.25663705,  1.36135682,  1.43117001,
                             1.53588977,  1.62315616,  1.71042264,  1.78023584,  1.88495561,
                             1.95476873,  2.05948857,  2.14675501,  2.23402144,  2.30383463,
                             2.40855435,  2.47836754,  2.58308727,  2.67035378,  2.75762025,
                             2.82743337,  2.93215314,  3.00196634,  3.1066861 , -3.08923282,
                             -3.00196634, -2.93215314, -2.82743337, -2.75762025, -2.65290041,
                             -2.56563397, -2.47836754, -2.40855435, -2.30383463, -2.23402144,
                             -2.12930171, -2.0420352 , -1.95476873, -1.88495561, -1.78023584,
                             -1.71042264, -1.60570288, -1.51843649, -1.43117001, -1.36135682,
                             -1.25663705, -1.18682393, -1.08210409, -0.99483765, -0.90757122,
                             -0.83775803, -0.7330383 , -0.66322511, -0.55850539, -0.47123887,
                             -0.3839724 , -0.31415928])

# Trolley
class TR:
    class probes:
        class position: # mm
            xy = np.array([
                [ 0.0,        0.0],
                [ 0.0,       -1.75],
                [ 1.75,       0.0],
                [ 0.0,        1.75],
                [-1.75,       0.0],
                [ 0.0,       -3.5],
                [ 1.75,      -3.031088],
                [ 3.0310889, -1.75],
                [ 3.5,        0.0],
                [ 3.0310889,  1.75],
                [ 1.75,       3.0310889],
                [ 0.0,        3.5],
                [-1.75,       3.0310889],
                [-3.031088,   1.75],
                [-3.5,        0.0],
                [-3.031088,  -1.75],
                [-1.75,      -3.031088]])*10.
            x     = xy[:,0]
            y     = xy[:,1]
            r     = np.sqrt(   xy[:,1]**2 + xy[:,0]**2)
            theta = np.arctan2(xy[:,1],     xy[:,0])
        n = len(position.x)
        grid = [(2,2),
                (3,2),
                (2,3),
                (1,2),
                (2,1),
                (4,2),
                (4,3),
                (3,4),
                (2,4),
                (1,4),
                (0,3),
                (0,2),
                (0,1),
                (1,0),
                (2,0),
                (3,0),
                (4,1)]

        class joePosition: #cm
            R1 = 1.75;  #cm
            R2 = 3.5; #cm
            ProbePosX = np.zeros([17])
            ProbePosY = np.zeros([17])
            ProbePosX[0] = 0.;
            ProbePosY[0] = 0.;
            ProbePosX[1] = 0.;
            ProbePosY[1] = -R1;
            ProbePosX[2] = R1;
            ProbePosY[2] = 0.;
            ProbePosX[3] = 0.;
            ProbePosY[3] = R1;
            ProbePosX[4] = -R1;
            ProbePosY[4] = 0.;
            ProbePosX[5] = R2*ROOT.TMath.Cos(9*ROOT.TMath.Pi()/6.);
            ProbePosY[5] = R2*ROOT.TMath.Sin(9*ROOT.TMath.Pi()/6.);
            ProbePosX[6] = R2*ROOT.TMath.Cos(10*ROOT.TMath.Pi()/6.);
            ProbePosY[6] = R2*ROOT.TMath.Sin(10*ROOT.TMath.Pi()/6.);
            ProbePosX[7] = R2*ROOT.TMath.Cos(11*ROOT.TMath.Pi()/6.);
            ProbePosY[7] = R2*ROOT.TMath.Sin(11*ROOT.TMath.Pi()/6.);
            ProbePosX[8] = R2*ROOT.TMath.Cos(12*ROOT.TMath.Pi()/6.);
            ProbePosY[8] = R2*ROOT.TMath.Sin(12*ROOT.TMath.Pi()/6.);
            ProbePosX[9] = R2*ROOT.TMath.Cos(13*ROOT.TMath.Pi()/6.);
            ProbePosY[9] = R2*ROOT.TMath.Sin(13*ROOT.TMath.Pi()/6.);
            ProbePosX[10] = R2*ROOT.TMath.Cos(14*ROOT.TMath.Pi()/6.);
            ProbePosY[10] = R2*ROOT.TMath.Sin(14*ROOT.TMath.Pi()/6.);
            ProbePosX[11] = R2*ROOT.TMath.Cos(15*ROOT.TMath.Pi()/6.);
            ProbePosY[11] = R2*ROOT.TMath.Sin(15*ROOT.TMath.Pi()/6.);
            ProbePosX[12] = R2*ROOT.TMath.Cos(16*ROOT.TMath.Pi()/6.);
            ProbePosY[12] = R2*ROOT.TMath.Sin(16*ROOT.TMath.Pi()/6.);
            ProbePosX[13] = R2*ROOT.TMath.Cos(17*ROOT.TMath.Pi()/6.);
            ProbePosY[13] = R2*ROOT.TMath.Sin(17*ROOT.TMath.Pi()/6.);
            ProbePosX[14] = R2*ROOT.TMath.Cos(18*ROOT.TMath.Pi()/6.);
            ProbePosY[14] = R2*ROOT.TMath.Sin(18*ROOT.TMath.Pi()/6.);
            ProbePosX[15] = R2*ROOT.TMath.Cos(19*ROOT.TMath.Pi()/6.);
            ProbePosY[15] = R2*ROOT.TMath.Sin(19*ROOT.TMath.Pi()/6.);
            ProbePosX[16] = R2*ROOT.TMath.Cos(20*ROOT.TMath.Pi()/6.);
            ProbePosY[16] = R2*ROOT.TMath.Sin(20*ROOT.TMath.Pi()/6.);

