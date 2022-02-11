import gtsam #https://github.com/borglab/gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from robot import meas_odom
from robot import meas_landmark

class solver:
    
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        
        self.initial_estimate = gtsam.Values()
        
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        #parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

        self.i = 0 #time index

    def updateisam(self):
        self.isam.update(self.graph, self.initial_estimate)

    def getResult(self):
        return self.isam.calculateEstimate()
           
    def addOdomMeasurement(self,meas: meas_odom):
        meas_x = meas.dr * np.cos(meas.dtheta)
        meas_y = meas.dr * np.sin(meas.dtheta)
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(meas.cov)
        
        factor = gtsam.BetweenFactorPose2(
                        X(self.i-1), X(self.i), 
                        gtsam.Pose2(meas_x, meas_y, meas.dtheta), odom_noise)
        self.graph.push_back(factor)

        self.initial_xi = pose.compose(noise)
        self.initial_estimate.insert(X(i), initial_xi)
    
    def addlandmarkMeasurement(self,meas: meas_landmark):
        rgbd_noise = gtsam.noiseModel(meas.cov)
        
        factor = gtsam.BearingRangeFactor2D(
                    X(self.i), L(meas.index),
                    meas.angle, meas.r, rgbd_noise)
        self.graph.push_back(factor)
