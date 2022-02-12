import gtsam #https://github.com/borglab/gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from robot import meas_odom
from robot import meas_landmark

class solver:
    
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph() #object 1
        self.initial_estimate = gtsam.Values() #object 2
        
        #object 3
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        #parameters.relinearizeSkip = 1 #doesnt work in my version of gtsam? should be setRelinearizeSkip? https://11187901375483992154.googlegroups.com/attach/3dde15f735496/sample.py?part=0.1&view=1&vt=ANaJVrGdeX_f8IFLF79358ZbSwCtgO6VOunOP2ZY6bSWzpjOHPOyvEvfcByKCCZDJm70YKtFyov_cxWbY67fsKT8XhhkAOSdHQ0VvoHdQ_EMAqR059Oh6XA
        self.isam2 = gtsam.ISAM2(parameters)

        #object 4
        self.i = 0 #time index

    def updateGraph(self):
        self.isam2.update(self.graph, self.initial_estimate)

    def updateTimeIndex(self):
        self.i += 1

    def getResult(self):
        return self.isam2.calculateEstimate()
           
    def addOdomMeasurement(self,meas: meas_odom):
        meas_x = meas.dr * np.cos(meas.dtheta)
        meas_y = meas.dr * np.sin(meas.dtheta)
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BetweenFactorPose2(
                        X(self.i-1), X(self.i), 
                        gtsam.Pose2(meas_x, meas_y, meas.dtheta), odom_noise)
        self.graph.push_back(factor)

        initial_Xi = self.pose.compose(gtsam.Pose2(meas.dx, meas.dy, meas.dtheta))
        self.initial_estimate.insert(X(self.i), initial_Xi)
    
    def addlandmarkMeasurement(self,meas: meas_landmark):
        rgbd_noise = gtsam.noiseModel.Gaussian.Covariance(meas.cov) #https://gtsam.org/doxygen/a03876.html
        
        factor = gtsam.BearingRangeFactor2D(
                    X(self.i), L(meas.index),
                    meas.angle, meas.range, rgbd_noise)
        self.graph.push_back(factor)
