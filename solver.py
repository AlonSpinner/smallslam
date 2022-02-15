import gtsam #https://github.com/borglab/gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from robot import meas_odom
from robot import meas_landmark
import utils

class solver:
    
    def __init__(self,X0 = None,X0cov = None, ax = None):
        if X0 is None:
            X0 = (0,0,0)
        if X0cov is None:
            X0cov = 0.01* np.eye(3)

        #initalize solver
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = self.initial_estimate

        #insert X0 to initial_estiamte and graph as prior
        X0 = gtsam.Pose2(X0)
        X0_prior_noise = gtsam.noiseModel.Gaussian.Covariance(X0cov)
        self.initial_estimate.insert(X(0), X0)
        self.graph.push_back(gtsam.PriorFactorPose2(X(0), X0, X0_prior_noise))

        #initalize solver isam2
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        #parameters.setRelinearizeSkip = 1 #doesnt work in my version of gtsam? should be setRelinearizeSkip? https://11187901375483992154.googlegroups.com/attach/3dde15f735496/sample.py?part=0.1&view=1&vt=ANaJVrGdeX_f8IFLF79358ZbSwCtgO6VOunOP2ZY6bSWzpjOHPOyvEvfcByKCCZDJm70YKtFyov_cxWbY67fsKT8XhhkAOSdHQ0VvoHdQ_EMAqR059Oh6XA
        self.isam2 = gtsam.ISAM2(parameters)

        #time index
        self.i = 0 

        #seen_landmarks to avoid initalizing landmarks twice (breaks solver)
        self.seen_landmarks = []

        #graphics
        self.ax = ax #might be None, depending on input
        self.graphics_landmarks = []
        self.graphics_poses = []

    def update(self, N=0):
        self.isam2.update(self.graph, self.initial_estimate)
        for kk in range(N):
            self.isam2.update() #can be called additional times to perform multiple optimizer iterations

        self.initial_estimate.clear() #inital_estimates only holds guesses for new variables.learnt from Pose2ISAM2Example.
        self.current_estimate = self.isam2.calculateEstimate()
           
    def addOdomMeasurement(self,meas: meas_odom):
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BetweenFactorPose2(
                        X(self.i-1), X(self.i), 
                        gtsam.Pose2(meas.dx, meas.dy, meas.dtheta), odom_noise)
        self.graph.push_back(factor)

        pose = self.current_estimate.atPose2(X(self.i-1))
        initial_Xi = pose.compose(gtsam.Pose2(meas.dx, meas.dy, meas.dtheta))
        self.initial_estimate.insert(X(self.i), initial_Xi)
    
    def addlandmarkMeasurement(self,meas: meas_landmark):
        rgbd_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BearingRangeFactor2D(
                    X(self.i), L(meas.index),
                    gtsam.Rot2.fromAngle(meas.angle), meas.range, rgbd_noise)
        self.graph.push_back(factor)

        if meas.index not in self.seen_landmarks: #then add landmark to inital_estimate and mark it as seen
            self.seen_landmarks.append(meas.index)
            pose = self.initial_estimate.atPose2(X(self.i))
            dx = meas.range * np.cos(pose.theta()+meas.angle)
            dy = meas.range * np.sin(pose.theta()+meas.angle)
            initial_L = gtsam.Point2(pose.x()+dx, pose.y()+dy)
            self.initial_estimate.insert(L(meas.index), initial_L)

    def plot_landmarks(self):
        if self.ax is None:
            raise TypeError("you must provide an axes handle to solver if you want to plot")

        marginals = gtsam.Marginals(self.graph, self.current_estimate)

        #remove old drawings if exist
        for graphics_lm in self.graphics_landmarks:
            for graphic in graphics_lm:
                    graphic.remove()

        self.graphics_landmarks = []
        for lm_index in self.seen_landmarks:
            cov = marginals.marginalCovariance(L(lm_index))
            loc = self.current_estimate.atPoint2(L(lm_index))
            self.graphics_landmarks.append(utils.plot_landmark(self.ax, loc = loc, cov = cov, color = 'b', marker = '.', markerSize = 1))

    def plot_poses(self,axis_length = 0.1):
        if self.ax is None:
            raise TypeError("you must provide an axes handle to solver if you want to plot")

        marginals = gtsam.Marginals(self.graph, self.current_estimate)

        #remove old drawings if exist
        for graphics_pose in self.graphics_poses:
                for graphic in graphics_pose:
                    graphic.remove()

        self.graphics_poses = []
        ii = 0
        while self.current_estimate.exists(X(ii)):
            cov = marginals.marginalCovariance(X(ii))
            pose = self.current_estimate.atPose2(X(ii))
            Rp2g = pose.rotation().matrix()  # rotation from pose to global
            origin = pose.translation()

            self.graphics_poses.append(utils.plot_pose(self.ax, Rp2g, origin, axis_length = axis_length, covariance = cov))
            ii +=1

