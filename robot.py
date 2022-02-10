import numpy as np
import matplotlib.pyplot as plt

class robot:
    def __init__(self,odometrey_noise = None, rgbd_noise = None):
        if odometrey_noise == None:
            odometrey_noise = np.zeros((2,2))
            odometrey_noise[0,0] = 0.1**2 #tangent noise
            odometrey_noise[1,1] =np.radians(3)**2 #angular noise

        if rgbd_noise == None:
            rgbd_noise = np.zeros((2,2))
            rgbd_noise[0,0] = 0.01**2 #depth noise
            rgbd_noise[1,1] = np.radians(1)**2 #angular noise

        self.pose = np.array([0.0,
                    0.0,
                    0.0]) # pose of robot [x,y,theta]
        self.FOV = np.radians(90.0)
        self.range = 1.0 #m
        self.odometrey_noise = odometrey_noise
        self.rgbd_noise = rgbd_noise

    def moveAndMeasureOdometrey(self,odom): 
        #odom = [dr,dtheta]
        #we formulate discrete state without velocities
        self.pose[2] += odom[1] #add dtheta 
        self.pose[0] += odom[0] * np.cos((self.pose[2])) #add dx
        self.pose[1] += odom[0] * np.sin((self.pose[2])) #add dy
        #note: we first rotate and than add translation. Same as in transform matrix. Oren agrees
        return self.odomModel(odom)
    
    def measureLandmarks(self,worldmap):
            #world - list of semanticly labled landmarks: [[L1_x,L1_y,L1_class],...]
            measurements = []
            for lm in worldmap.landmarks:
                    meas = self.rgbdMeasModel(lm)
                    if meas: #if measurement took place
                        measurements.append(meas)
            return measurements

    def rgbdMeasModel(self,lm):
        gt_dx , gt_dy = lm.x-self.pose[0], lm.y-self.pose[1]
        gt_r = (gt_dx**2 + gt_dy**2)**0.5
        gt_angle = np.arctan2(gt_dy,gt_dx)
        if gt_angle < self.FOV/2 and (gt_r < self.range): #if viewed, compute noisy measurement
            mu = np.array([gt_r,gt_angle]).squeeze() #must be 1D numpy array
            dr, dangle = np.random.multivariate_normal(mu, self.rgbd_noise) 
            return meas_landmark(dr,dangle,lm.classLabel,lm.index)

    def odomModel(self,odom):
        meas_dr, meas_dtheta = np.random.multivariate_normal(odom, self.odometrey_noise)
        return meas_odom(meas_dr,meas_dtheta) #dr,dtheta. same format as input

class meas_landmark:
    # data container
    def __init__(self,r = 0, angle = 0, classLabel = 'clutter', index = []):
        self.r = r
        self.angle = angle
        self.classLabel = classLabel
        self.index = index

class meas_odom:
    # data container
    def __init__(self, dr = 0, dtheta = 0):
            self.dr = dr
            self.dtheta = dtheta

    def toNumpy(self):
        return np.array([self.dr,self.dtheta].squeeze())

    

