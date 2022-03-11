import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import gtsam
from gtsam.symbol_shorthand import L, X
import graphviz
import os

import smallslam.utils.plotting as plotting
from smallslam.solver import solver
from smallslam.robot import robot
from smallslam.map import map

def scenario():
    np.random.seed(seed=3) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-5,+25); yrange = (-5,+25)
    fig , (axWorld,axError) = plotting.spawnWorld(xrange, yrange, type = "world and error")
    
    #------landmarks
    N = 10
    semantics = ("table","chair")
    worldMap = map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange,sigmarange = (-0.5,0.5))
    worldMap.plot(ax = axWorld, plotIndex = True,plotCov = False)

    #------Spawn Robot
    odometry_noise = np.zeros((3,3))
    odometry_noise[0,0] = 0.5**2 #dx noise
    odometry_noise[1,1] = 0.5**2 #dy noise
    odometry_noise[2,2] =np.radians(5)**2 #angular noise

    rgbd_noise = np.zeros((2,2))
    rgbd_noise[0,0] = 0.1**2 #depth noise
    rgbd_noise[1,1] = np.radians(1)**2 #angular noise

    car = robot(ax = axWorld, FOV = np.radians(90), markerSize = 50, 
                      odometry_noise = odometry_noise, rgbd_noise = rgbd_noise)
    car.range = 10
    
    #------ ground truth odometrey
    a = 20 #square side length
    rotSteps = 5
    linSteps = 8
    odomLine = [gtsam.Pose2(a/linSteps,0,0)] * linSteps
    odomTurn = [gtsam.Pose2(0,0,np.pi/2/rotSteps)] * rotSteps
    
    odom = []
    for _ in range(5):
        odom = odom + odomLine + odomTurn 

    return car, worldMap, axWorld, axError, fig, odom


car, worldMap, axWorld, axError,  fig, gt_odom = scenario()
backendX0 = gtsam.Pose2(0,5,0)
backend = solver(ax = axWorld, X0 = backendX0 ,X0cov = car.odometry_noise/1000, semantics = worldMap.exportSemantics())

#init history loggers
hist_GT, hist_DR, hist_ISAM2 = car.pose.translation(), car.pose.translation(), backendX0.translation()
hist_DR_error, hist_ISAM2_error = np.nan, np.nan

#set graphics
graphic_GT_traj, = axWorld.plot([], [],'ko-',markersize = 3)
graphic_DR_traj, = axWorld.plot([], [],'ro-',markersize = 3)
graphic_ISAM2_traj, = axWorld.plot([], [],'bo-',markersize = 3)
graphics_DR_error, = axError.plot([], [], 'ro-', markersize = 3)
graphics_ISAM2_error, = axError.plot([], [], 'bo-', markersize = 3)

fig.legend(handles = [graphic_GT_traj,graphic_DR_traj,graphic_ISAM2_traj], 
        labels = ["ground truth","dead reckoning","SLAM solver"])

#run and plot simulation and provde movie
moviewriter = PillowWriter(fps = 5)
moviewriter.setup(fig,'05_movie.gif',dpi = 100)
xcurrent_DR = car.pose
with plt.ion():
    for kk, odom in enumerate(gt_odom):
        meas_odom = car.moveAndMeasureOdometrey(odom)
        meas_lms = car.measureLandmarks(worldMap.landmarks)
            
        backend.i += 1 #increase time index. Must be done before adding measurements as factors
        backend.addOdomMeasurement(meas_odom)
        for meas_lm in meas_lms:
            backend.addlandmarkMeasurement(meas_lm)
        backend.update(N=0)

        if kk == 20:
            backend.graph.remove(0)
            backend.addlandmarkPrior(worldMap.landmarks[7])
            backend.addlandmarkPrior(worldMap.landmarks[5])
            backend.addlandmarkPrior(worldMap.landmarks[1])
            axWorld.set_title(f"registration to map: landmarks 1,5,7 \n"
                                "optimizing on full state")
            backend.optimizeFullState()

            for _ in np.arange(0.0, 1.0, 0.1):
                moviewriter.grab_frame()
                plt.pause(0.1)

        #dead reckoning integration
        xcurrent_DR = xcurrent_DR.compose(meas_odom.dpose)

        xcurrent_ISAM2 = backend.isam2.calculateEstimatePose2(X(backend.i))

        #log history
        hist_GT = np.vstack([hist_GT,car.pose.translation()])
        hist_DR = np.vstack([hist_DR,xcurrent_DR.translation()])
        hist_ISAM2 = np.vstack([hist_ISAM2,xcurrent_ISAM2.translation()])
        hist_DR_error = np.vstack([hist_DR_error,np.linalg.norm(xcurrent_DR.translation()-car.pose.translation())])
        hist_ISAM2_error = np.vstack([hist_ISAM2_error,np.linalg.norm(xcurrent_ISAM2.translation()-car.pose.translation())])

        #plot and update
        car.plot()
        axWorld.set_title(f"step {kk}")
        backend.plot(poses = True, landmarks = False, poses_Cov = True, poses_axis_length = 1.0)

        graphic_GT_traj.set_data(hist_GT[:,0],hist_GT[:,1])
        graphic_DR_traj.set_data(hist_DR[:,0],hist_DR[:,1])
        graphic_ISAM2_traj.set_data(hist_ISAM2[:,0],hist_ISAM2[:,1])
        graphics_DR_error.set_data(range(len(hist_DR_error)),hist_DR_error)
        graphics_ISAM2_error.set_data(range(len(hist_ISAM2_error)),hist_ISAM2_error)
        axError.relim()
        axError.autoscale_view(True,True,True)

        moviewriter.grab_frame()
        plt.pause(0.1)

#save graph as pdf in folder 
graphName = '05_graph'
backend.graph.saveGraph(graphName) #temp file
graphviz.render('dot','pdf',graphName) #creates PDF
os.remove(graphName) #delete temp file

moviewriter.finish()