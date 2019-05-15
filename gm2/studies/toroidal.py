import gm2

def legendreQ(m,n,z):
    """ LegendreQ(m-1/2,n,cosh(x)) """

def DlegendreQ(m,n,z):
    """ derivative of LegendreQ(m-1/2,n,cosh(x)) """

def zeta(x_center, y, r0):
    """ r, z -> zeta """
    return np.arctanh(2.*x_center*r0/(x_center**2 + r0**2 + y**2))

def eta(x_center, y, r0):
    """ r, z -> eta """
    xx = 2.*r0*y / (x_center**2 - r0**2 + y**2);
    if(np.abs(xx)<0.001):
        etax = 1.
        for i in range(10):
            etax += xx**(2.*i)/(2.*i+1.)*np.cos(np.pi * i)
        etax = etax * xx;
    else:
        etax = np.arctan(xx)
    if x_center < np.sqrt(r0**2):
        etax = etax + np.pi
    return etax
