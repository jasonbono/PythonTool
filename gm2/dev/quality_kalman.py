# coding: utf-8
import gm2
import numpy as np
import matplotlib.pyplot as plt


tr = gm2.Trolley([3997])
def callback():
    return [tr.getTimeGPS(), tr.getPhi(), tr.getFrequency(), tr.getFidLength(), tr.getAmplitude()]
time, phi, freq, length, amp = tr.loop(callback)

probe = 0
skip = 200

# Kalman filter
# X = [f, df/ds]
er_est_f  =  5
er_est_df = 10
er_m_f    =  5
er_m_df   = 10



def predict(f, df, s, a):
    
    A = np.array([[1, a],
                  [0, 0]])
    X = np.array([[f],
                  [df]])
    B = np.array([[0.5 * s**2], # dff * 1/2 * s^2
                  [s]])          # df * s
    x_ = A.dot(X) + B.dot(a)
    #x_ = np.array([[f + df * s],
    #               [df]])
    return x_

def covariance(s1, s2):
    cov = np.array([[s1**2, s1*s2],
                    [s2*s1, s2**2]])
    return np.diag(np.diag(cov))


t = phi[skip+1,probe,0] - phi[skip,probe,0]

P = np.array([[er_est_f**2, 0],
              [0, er_est_df**2]])
A = np.array([[1, t],
              [1, 0]])
X = np.array([[freq[skip,probe,0]],
              [freq[skip+1,probe,0] - freq[skip,probe,0]]])

yy = np.ones(freq.shape[0])
for i in np.arange(skip+1, freq.shape[0]-skip):
    # Prediction
    t = phi[i+1,probe,0] - phi[i,probe,0]
    A = np.array([[1, t],
                  [1, 0]])
    X = predict(X[0][0], X[1][0], t, 20)
    #print(t, X)
    P = np.diag(np.diag(A.dot(P).dot(A.T)))
    
    # Calculating the Kalman Gain
    H = np.identity(2)
    R = covariance(er_m_f, er_m_df)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H).dot(np.linalg.inv(S))

    tl = phi[i,probe,0] - phi[i-1,probe,0]
    if tl == 0:
        Y = np.array([[freq[i, probe, 0]], [0]])
    else:
        Y = np.array([[freq[i, probe, 0]], [(freq[i, probe, 0] - freq[i-1, probe, 0])/tl]])
    #print(Y,X)
    X = X + K.dot(Y - H.dot(X))
    yy[i] = X[0][0]
    P = (np.identity(len(K)) - K.dot(H)).dot(P)
    #print(X[0][0]-freq[i,0,0])i
plt.plot(phi[skip:,0,0], freq[skip:,0,0],'.')
plt.plot(phi[skip:,0,0], yy[skip:],'--')
plt.show()
