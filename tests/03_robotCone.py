import numpy as np
from smallslam.robot import robot
import smallslam.utils.plotting as plotting
import gtsam
import matplotlib.pyplot as plt

np.random.seed(seed=0)

#------Build worldmap
fig , ax = plotting.spawnWorld(xrange = (-2,2), yrange = (-2,2))

#------Spawn Robot
pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
car = robot(pose = pose0)
car.plot(ax)

#control inputs - taken from unitTest0 relative odometrey
r = 1; m = 20
dx = 0.309017 ;dy = 0.0489435; dtheta = 0.314159
gt_odom = [gtsam.Pose2(dx,dy,dtheta)] * 2*m

with plt.ion():
    for odom in gt_odom:
        car.moveAndMeasureOdometrey(odom)

        car.plot(ax)
        plt.pause(0.1)