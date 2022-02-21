import gtsam #https://github.com/borglab/gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from robot import meas_odom
from robot import meas_landmark
import utils
import map

class solver:
    
    def __init__(self,X0 = None,X0cov = None, semantics = None ,ax = None):
        if X0 is None:
            X0 = gtsam.Pose2((0,0,0))
        if X0cov is None:
            X0cov = 0.001* np.eye(3)

        #initalize solver
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = self.initial_estimate

        #insert X0 to initial_estiamte and graph as prior
        X0_prior_noise = gtsam.noiseModel.Gaussian.Covariance(X0cov)
        self.initial_estimate.insert(X(0), X0)
        self.graph.push_back(gtsam.PriorFactorPose2(X(0), X0, X0_prior_noise))

        #initalize solver isam2
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.setRelinearizeSkip(1)
        self.isam2 = gtsam.ISAM2(parameters)

        #time index
        self.i = 0 

        #seen_landmarks to avoid initalizing landmarks twice (breaks solver)
        self.seen_landmarks = {
                             'index': [],
                             'classLabel': []
                              }

        #for extensive plotting or data assosication
        #exported from map object. dictionary with following keys: "classLabel","color","marker", each holding a list, representing tabular data
        self.semantics = semantics 

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
                        meas.dpose, odom_noise)
        self.graph.push_back(factor)

        pose = self.current_estimate.atPose2(X(self.i-1))
        initial_Xi = pose.compose(meas.dpose)
        self.initial_estimate.insert(X(self.i), initial_Xi)
    
    def addlandmarkMeasurement(self,meas: meas_landmark):
        rgbd_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BearingRangeFactor2D(
                    X(self.i), L(meas.index),
                    gtsam.Rot2.fromAngle(meas.angle), meas.range, rgbd_noise)
        self.graph.push_back(factor)

        if meas.index not in self.seen_landmarks["index"]: #then mark it as seen and it add landmark to inital_estimate
            
            self.seen_landmarks["index"].append(meas.index)
            self.seen_landmarks["classLabel"].append(meas.classLabel)

            pose = self.initial_estimate.atPose2(X(self.i))
            xy_ego = meas.range * np.array([np.cos(meas.angle),np.sin(meas.angle)])
            xy_world = pose.transformFrom((xy_ego))
            initial_L = gtsam.Point2(xy_world)

            self.initial_estimate.insert(L(meas.index), initial_L)

    def addlandmarkPrior(self, lm: map.landmark):
        L_prior_noise = gtsam.noiseModel.Gaussian.Covariance(lm.cov) #https://gtsam.org/doxygen/a03876.html
        Li = gtsam.Point2(lm.xy)
        self.graph.push_back(gtsam.PriorFactorPoint2(L(lm.index), 
                                                     Li, 
                                                     L_prior_noise))
 
        if lm.index not in self.seen_landmarks["index"]:
            self.seen_landmarks["index"].append(lm.index)
            self.seen_landmarks["classLabel"].append(lm.classLabel)
            self.initial_estimate.insert(L(lm.index), Li)

    def plot_landmarks(self,plotIndex = False, plotSemantics = False):
        if self.ax is None:
            raise Exception("you must provide an axes handle to solver if you want to plot")
        if plotSemantics is True and self.semantics is None:
            raise Exception("You must provide semantics if you want to plot them. Currently empty.")

        marginals = gtsam.Marginals(self.graph, self.current_estimate)

        #remove old drawings if exist
        for graphics_lm in self.graphics_landmarks:
            for graphic in graphics_lm:
                    graphic.remove()

        self.graphics_landmarks = []
        for seen_lm_index, seen_lm_classLabel in zip(self.seen_landmarks["index"],self.seen_landmarks["classLabel"]):

            #prep for plots
            index4plot = None if plotIndex is False else seen_lm_index
            if self.semantics is not None and plotSemantics is True:
                ii = self.semantics["classLabel"].index(seen_lm_classLabel) #find index of class label in list (this needs to be a hashmap)
                markerShape = self.semantics['marker'][ii]
            else:
                markerShape = '.' #default

            #plot
            cov = marginals.marginalCovariance(L(seen_lm_index))
            loc = self.current_estimate.atPoint2(L(seen_lm_index))
            self.graphics_landmarks.append(utils.plot_landmark(self.ax, loc = loc, cov = cov,
                markerColor = 'b', markerShape = markerShape, markerSize = 3,
                index = index4plot, textColor = 'b'))

    def plot_poses(self,axis_length = 0.1, plotCov = True):
        if self.ax is None:
            raise Exception("you must provide an axes handle to solver if you want to plot")

        marginals = gtsam.Marginals(self.graph, self.current_estimate)

        #remove old drawings if exist
        for graphics_pose in self.graphics_poses:
                for graphic in graphics_pose:
                    graphic.remove()

        self.graphics_poses = []
        ii = 0
        while self.current_estimate.exists(X(ii)):
            cov = marginals.marginalCovariance(X(ii)) if plotCov is True else None
            pose = self.current_estimate.atPose2(X(ii))

            self.graphics_poses.append(utils.plot_pose(self.ax, pose, axis_length = axis_length, covariance = cov))
            ii +=1
