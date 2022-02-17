import numpy as np

def R2(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta),np.cos(theta)]])

def relativeTransform(Ta2w,Tb2w):
    Tw2b = inverseTransform(Tb2w)
    Ta2b = Tw2b @ Ta2w
    return Ta2b

def inverseTransform(Ta2b):
    #Ta2b = [Ra2b,t_b_b2a ; 0 0 1] - size 3x3
    Ra2b, t_b_b2a = T2Rt(Ta2b)
    Rb2a = Ra2b.T
    t_a_a2b = Rb2a @ (-t_b_b2a)
    Tb2a = Rt2T(Rb2a,t_a_a2b)
    return Tb2a

def Rt2T(R,t):
    M2x3 = np.hstack([R,t])
    M1x3 = np.array([[0, 0, 1]])
    return np.vstack([M2x3,M1x3])

def T2Rt(T):
    R = T[:2,:2]
    t = T[:2,[2]] #having [2] instead of just 2 keeps the 2d dimensions of the vector
    return R,t

def odomTFromTrajT(T):
    dT = []
    for ii in range(1,len(T)):
        dT.append(relativeTransform(T[ii-1],T[ii]))
    return dT

def Te2w_to_pose(Te2w):
    v = Te2w[:2,0] #[cos,sin]
    theta = np.arctan2(v[1],v[0])
    x = Te2w[0,2]
    y = Te2w[1,2]
    return x, y, theta