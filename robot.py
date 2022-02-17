import numpy as np
import matplotlib.pyplot as plt
import utils
import gtsam
import transforms2D

class robot:
    def __init__(self,odometry_noise = None, rgbd_noise = None,
                 FOV = 60.0*np.pi/180.0, range = 1.0, pose: gtsam.Pose2 = None, 
                 ax = None, markerSize=200):

        #input assumptions
        if odometry_noise == None:
            odometry_noise = np.zeros((3,3))
            odometry_noise[0,0] = 0.1**2 #dx noise
            odometry_noise[1,1] = 0.1**2 #dy noise
            odometry_noise[2,2] =np.radians(3)**2 #angular noise

        if rgbd_noise == None:
            rgbd_noise = np.zeros((2,2))
            rgbd_noise[0,0] = 0.01**2 #depth noise
            rgbd_noise[1,1] = np.radians(1)**2 #angular noise

        #starting location
        if pose is None: #not set in defaults as numpy arrays are mutable! it causes the class to rememmber inital value
            pose = gtsam.Pose2(np.eye(3))
        self.pose = pose
        
        #sensor physics
        self.FOV = FOV #radians
        self.range = range #[m]
        self.odometry_noise = odometry_noise
        self.rgbd_noise = rgbd_noise
        
        #place holder for graphic handles
        self.graphic_rgbd = []
        self.graphic_car = []

        if ax is not None:
            self.plot(ax, markerSize = markerSize)

    def moveAndMeasureOdometrey(self,odom: gtsam.Pose2): 
        self.pose = self.pose.compose(odom)
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
        gt_lm_ego = self.world2Ego(np.vstack((lm.x,lm.y)))

        gt_angle = np.arctan2(gt_lm_ego[1],gt_lm_ego[0])[0] #np.arctan2 returns np.array when I want it to be float for later
        gt_r = np.linalg.norm(gt_lm_ego)

        if (gt_angle > -self.FOV/2 and gt_angle < +self.FOV/2) \
             and (gt_r < self.range): #if viewed, compute noisy measurement
            mu = np.array([gt_r,gt_angle])
            meas_dr, meas_dangle = np.random.multivariate_normal(mu, self.rgbd_noise) 
            return meas_landmark(meas_dr,meas_dangle,lm.classLabel,lm.index, self.rgbd_noise)

    def odomModel(self,odom):
        meas_dx, meas_dy, meas_dtheta = np.random.multivariate_normal(utils.pose2ToNumpy(odom), self.odometry_noise)
        meas_odom = gtsam.Pose2(meas_dx,meas_dy,meas_dtheta)
        return meas_odom

    def plot(self,ax = None, markerSize = 200):
        #first call to plot should have ax variable included, unless you want to open a new axes.
        if ax == None and not self.graphic_car: #no ax given, and car was not plotted before
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()

        phi = np.linspace(-self.FOV/2,self.FOV/2,10)
        p = self.range * np.array([np.cos(phi),np.sin(phi)])
        xy = np.vstack((self.pose[:2],self.ego2World(p).T))

        if self.graphic_car: #car was plotted before
            self.graphic_car.set_offsets(self.pose[:2]) #https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
            self.graphic_rgbd.set_xy(xy)  #https://stackoverflow.com/questions/38341722/animation-to-translate-polygon-using-matplotlib
        else:
            self.graphic_car = ax.scatter(self.pose[0],self.pose[1],s = markerSize, c = 'k', marker = 'o')
            self.graphic_rgbd, = ax.fill(xy[:,0],xy[:,1], facecolor = "b" , alpha=0.1, animated = False)

    def world2Ego(self,xyWorld):
        #accepts 2XM and return 2XM
        Rw2e = transforms2D.R2(-self.pose[2])
        t_e_e2w = Rw2e @ (-np.atleast_2d(self.pose[:2]).T)

        xyEgo = Rw2e @ xyWorld + t_e_e2w
        return xyEgo

    def ego2World(self,xyEgo):
        #accepts 2XM and return 2XM
        Re2w = transforms2D.R2(self.pose[2])
        t_w_w2e = np.atleast_2d(self.pose[:2]).T
        xyWorld = Re2w @ xyEgo + t_w_w2e
        return xyWorld
        

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
    

