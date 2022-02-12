import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

class robot:
    def __init__(self,odometrey_noise = None, rgbd_noise = None, ax = None):
        #input assumptions
        if odometrey_noise == None:
            odometrey_noise = np.zeros((2,2))
            odometrey_noise[0,0] = 0.1**2 #tangent noise
            odometrey_noise[1,1] =np.radians(3)**2 #angular noise

        if rgbd_noise == None:
            rgbd_noise = np.zeros((2,2))
            rgbd_noise[0,0] = 0.01**2 #depth noise
            rgbd_noise[1,1] = np.radians(1)**2 #angular noise

        #starting location
        self.pose = np.array([0.0,
                    0.0,
                    0.0]) # pose of robot [x,y,theta]
        #sensor physics
        self.FOV = np.radians(60.0)
        self.range = 1.0 #m
        self.odometrey_noise = odometrey_noise
        self.rgbd_noise = rgbd_noise
        
        #place holder for graphic handles
        self.graphic_rgbd = []
        self.graphic_car = []

        if ax is not None:
            self.plot(ax)

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
        meas_dx, meas_dy, meas_dtheta = np.random.multivariate_normal(odom, self.odometrey_noise)
        return meas_odom(meas_dx,meas_dy,meas_dtheta)

    def plot(self,ax = None, plotCov = False):
        #first call to plot should have ax variable included, unless you want to open a new axes.
        if ax == None and not self.graphic_car: #no ax given, and car was not plotted before
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()

        phi = np.linspace(self.pose[2]-self.FOV/2,self.pose[2]+self.FOV/2,10)
        p = self.range * np.array([np.cos(phi),np.sin(phi)]).T + self.pose[:2]
        xy = np.vstack((self.pose[:2],p))

        if self.graphic_car: #car was plotted before
            self.graphic_car.set_offsets(self.pose[:2]) #https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
            self.graphic_rgbd.set_xy(xy)  #https://stackoverflow.com/questions/38341722/animation-to-translate-polygon-using-matplotlib
        else:
            self.graphic_car = ax.scatter(self.pose[0],self.pose[1],s = 200, c = 'k', marker = 'o')
            self.graphic_rgbd, = ax.fill(xy[:,0],xy[:,1], facecolor = "b" , alpha=0.1, animated = False)

class meas_landmark:
    # data container
    def __init__(self,range = 0, angle = 0, classLabel = 'clutter', index = [], cov = []):
        self.range = range
        self.angle = angle
        self.classLabel = classLabel
        self.index = index
        self.cov = cov

class meas_odom:
    # data container
    def __init__(self, dx = 0, dy = 0, dtheta = 0, cov = []):
            self.dx = dx
            self.dy = dy
            self.dtheta = dtheta
            self.cov = cov

    def toNumpy(self):
        return np.array([self.dr,self.dtheta].squeeze())
    

