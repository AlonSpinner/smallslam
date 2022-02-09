import numpy as np

class robot:
    def __init__(self,odometrey_noise = None, rgbd_noise = None):
        if odometrey_noise == None:
            odometrey_noise = np.zeros((2,2))
            odometrey_noise[0,0] = 0.1**2
            odometrey_noise[1,1] =np.radians(3)**2

        if rgbd_noise == None:
            rgbd_noise = np.zeros((2,2))
            rgbd_noise[0,0] = 0.01**2
            rgbd_noise[1,1] = np.radians(1)**2

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

        meas_odom = np.random.multivariate_normal(odom, self.odometrey_noise)
        return meas_odom #dr,dtheta. same format as input
    
    def measureLandmarks(self,world):
            #world - list of semanticly labled landmarks: [[L1_x,L1_y,L1_class],...]
            measurement = []
            for lm in world:
                gt_dx , gt_dy = lm[0]-self.pose[0], lm[1]-self.pose[1]
                gt_r = (gt_dx**2 + gt_dy**2)**0.5
                gt_angle = np.arctan2(gt_dy,gt_dx)
                if gt_angle < self.FOV/2 and (gt_r < self.range):
                    meas_r, meas_angle = self.rgbdMeasModel(gt_r,gt_angle)
                    measurement.append([meas_r,meas_angle,lm[2]]) #lm[2] is the class of lm
            return measurement

    def rgbdMeasModel(self,gt_r,gt_angle):
        meas_rgbd = np.random.multivariate_normal((gt_r,gt_angle), self.odometrey_noise)
        return meas_rgbd
    

