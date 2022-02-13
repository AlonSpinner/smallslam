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
        #parameters.relinearizeSkip = 1 #doesnt work in my version of gtsam? should be setRelinearizeSkip? https://11187901375483992154.googlegroups.com/attach/3dde15f735496/sample.py?part=0.1&view=1&vt=ANaJVrGdeX_f8IFLF79358ZbSwCtgO6VOunOP2ZY6bSWzpjOHPOyvEvfcByKCCZDJm70YKtFyov_cxWbY67fsKT8XhhkAOSdHQ0VvoHdQ_EMAqR059Oh6XA
        self.isam2 = gtsam.ISAM2(parameters)

        X0 = gtsam.Pose2(0,0,0)
        X0_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        # prior on X(0) and initial estimate for it
        self.initial_estimate.insert(X(0), X0)
        self.graph.push_back(gtsam.PriorFactorPose2(X(0), X0, X0_prior_noise))

        self.i = 0 #time index
        self.update() #evaluates current_estimate from priors and adds +1 to i

    def update(self):
        self.isam2.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam2.calculateEstimate()
        self.i += 1
           
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
