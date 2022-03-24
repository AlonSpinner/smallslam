from typing import overload
from smallslam.map import map
from smallslam.utils.datatypes import landmark_mesh
from smallslam.utils.plotting import spawnWorld
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from smallslam.solver import solver
from smallslam.robot import robot
import gtsam
from gtsam.symbol_shorthand import L, X
from sgicp.clusters import cluster2
from sgicp.sgicp import semantic_gicp
import sgicp

def buildMaps(plot=False):

    modelMap = map()
    meshes = []

    #---------------------WALLS
    id = 1
    p1 = np.array([0.5,0]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([2,0]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 2
    p1 = np.array([2,0]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([4,0]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 3
    p1 = np.array([2,0]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([2,1.5]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 4
    p1 = np.array([4,0]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([4,2]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 5
    p1 = np.array([4,2]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([1.5,2]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 6
    p1 = np.array([4,2]); cov1 = np.array([[0.00820,-0.00360],[-0.00360,0.00280]])
    p2 = np.array([2,3]); cov2 = np.array([[0.00820,-0.00360],[-0.00360,0.00280]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))


    id = 7
    p1 = np.array([2,2.5]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([2,3]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(5))


    id = 8
    p1 = np.array([0,3]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([2,3]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))

    id = 9
    p1 = np.array([0,3]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([0,2]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(5))


    id = 10
    p1 = np.array([0,2]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([1,2]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(5))


    id = 11
    p1 = np.array([0,2]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([0,0]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'wall'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))


    #---------------------DOORS
    id = 12
    p1 = np.array([0,0]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([0.5,0]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'door'))
    modelMap.addLandmarks(meshes[-1].interpolate(4))


    id = 13
    p1 = np.array([1,2]); cov1 = np.array([[0.01,0],[0,0.001]])
    p2 = np.array([1.5,2]); cov2 = np.array([[0.01,0],[0,0.001]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'door'))
    modelMap.addLandmarks(meshes[-1].interpolate(4))


    id = 14
    p1 = np.array([2,1.5]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([2,2]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'door'))
    modelMap.addLandmarks(meshes[-1].interpolate(4))


    id = 15
    p1 = np.array([2,2.5]); cov1 = np.array([[0.001,0],[0,0.01]])
    p2 = np.array([2,2]); cov2 = np.array([[0.001,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.array([0,1]),
                        xy = np.array([p1,p2]),
                        cov = np.array([cov1,cov2]),
                        classLabel = 'door'))
    modelMap.addLandmarks(meshes[-1].interpolate(4))


    #---------------------FURNITURE

    id = 16
    p1 = np.array([2.5,0.5]); cov1 = np.array([[0.01,0],[0,0.01]])
    p2 = np.array([3.5,0.5]); cov2 = np.array([[0.01,0],[0,0.01]])
    p3 = np.array([3.5,1.5]); cov3 = np.array([[0.01,0],[0,0.01]])
    p4 = np.array([2.5,1.5]); cov4 = np.array([[0.01,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.linspace(0,1,5),
                        xy = np.array([p1,p2,p3,p4,p1]),
                        cov = np.array([cov1,cov2,cov3,cov4,cov1]),
                        classLabel = 'furniture'))
    modelMap.addLandmarks(meshes[-1].interpolate(30))


    id = 17
    p1 = np.array([0.5,2.25]); cov1 = np.array([[0.01,0],[0,0.01]])
    p2 = np.array([0.75,2.25]); cov2 = np.array([[0.01,0],[0,0.01]])
    p3 = np.array([0.75,2.5]); cov3 = np.array([[0.01,0],[0,0.01]])
    p4 = np.array([0.5,2.5]); cov4 = np.array([[0.01,0],[0,0.01]])
    meshes.append(landmark_mesh(id = id, 
                        u = np.linspace(0,1,5),
                        xy = np.array([p1,p2,p3,p4,p1]),
                        cov = np.array([cov1,cov2,cov3,cov4,cov1]),
                        classLabel = 'furniture'))
    modelMap.addLandmarks(meshes[-1].interpolate(10))   

    #---------------------OBSERVABLE MAP

    observeableMap = copy.deepcopy(modelMap)
    lms = observeableMap.landmarks
    random.shuffle(lms)
    observeableMap.landmarks = lms[::]
    observeableMap.fillMapRandomly(N = 20, classLabels = ['clutter'], xrange = (0.0,4.0), yrange=(0.0,3.0), sigmarange = (0.001,0.1))
    
    if plot:
        ax = modelMap.plot(plotCov = True, plotLegend = True, plotMeshIndex = True)
        ax.set_title('Model map')    
        
        ax = observeableMap.plot(plotCov = True, plotLegend = True, plotMeshIndex = True)
        ax.set_title('observable map')
        
        plt.show()

    return modelMap, observeableMap
def buildRobotAndOdom():
    odometry_noise = np.zeros((3,3))
    odometry_noise[0,0] = 0.01**2 #dx noise
    odometry_noise[1,1] = 0.01**2 #dy noise
    odometry_noise[2,2] =np.radians(3)**2 #angular noise

    rgbd_noise = np.zeros((2,2))
    rgbd_noise[0,0] = 0.01**2 #depth noise
    rgbd_noise[1,1] = np.radians(1)**2 #angular noise

    car = robot(pose = gtsam.Pose2(0.5,2.75,0), FOV = np.radians(90),range = 1,
                        odometry_noise = odometry_noise, rgbd_noise = rgbd_noise)
    odomForward = [gtsam.Pose2(0.1,0,0)]
    odomTurnLeft = [gtsam.Pose2(0,0,np.pi/8)]
    odomTurnRight = [gtsam.Pose2(0,0,-np.pi/8)]
    odom = odomForward*8 + odomTurnRight*4 + odomForward*10 + odomTurnLeft*4 + odomForward*20
    return car, odom
def backendToCluster2(backend):
    current_estimate = backend.calculateEstimate(bestEstimate = True)
    marginals = gtsam.Marginals(backend.graph, current_estimate)
    cov = []; loc = []
    for seen_lm_index, seen_lm_classLabel in zip(backend.seen_landmarks["id"],backend.seen_landmarks["classLabel"]):
        cov.append(marginals.marginalCovariance(L(seen_lm_index)))
        loc.append(current_estimate.atPoint2(L(seen_lm_index)))

    cluster = cluster2()
    cluster.addPoints(np.array(loc).reshape(-1,2,1),np.array(cov),backend.seen_landmarks["classLabel"])
    return cluster
def mapToCluster2(m):
    loc = []; cov = []; classLabels = []
    for lm in m.landmarks:
        loc.append(lm.xy)
        cov.append(lm.cov)
        classLabels.append(lm.classLabel)
    cluster = cluster2()
    cluster.addPoints(np.array(loc).reshape(-1,2,1),np.array(cov),classLabels)
    return cluster

modelMap, observeableMap = buildMaps(plot = False)
car,gt_odom = buildRobotAndOdom()

_, axSolver = spawnWorld()
_, axWorld = spawnWorld()
backend = solver(ax = axSolver, X0 = gtsam.Pose2(0,2.5,0) ,X0cov = car.odometry_noise/1000, semantics = observeableMap.exportSemantics())

observeableMap.plot(ax = axWorld)
with plt.ion():
    for kk, odom in enumerate(gt_odom):
        meas_odom = car.moveAndMeasureOdometrey(odom)
        meas_lms = car.measureLandmarks(observeableMap.landmarks)
            
        backend.i += 1 #increase time index. Must be done before adding measurements as factors
        backend.addOdomMeasurement(meas_odom)
        for meas_lm in meas_lms:
            backend.addlandmarkMeasurement(meas_lm)
        backend.update(N=0)

        backend.plot(landmarks_Semantics = True)

        car.plot(ax = axWorld, markerSize = 10)
        
        plt.pause(0.1)

backendCluster = backendToCluster2(backend)
mapCluster = mapToCluster2(modelMap)

invx_est, fmin, itr, df, i =  semantic_gicp(backendCluster.points, backendCluster.covariances, backendCluster.pointLabels, #source
                                        mapCluster.points,mapCluster.covariances, mapCluster.pointLabels, #target
                                        n_iter_max = 50,
                                        x0 = np.zeros(3),
                                        n = 2,
                                        tol = 1e-6,
                                        solverloss = 'cauchy', solver_f_scale = 2)

ax, _ = sgicp.utils.plotting2D.prepAxes()
backendCluster.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b')
mapCluster.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r')
ax.set_title('before ICP')

x_est = sgicp.utils.transforms2D.inverse_x(invx_est)
invEst_R, invEst_t = sgicp.utils.transforms2D.x_to_Rt(invx_est)
backendCluster.transform(invEst_R,invEst_t, transformCov = True)

ax, _ = sgicp.utils.plotting2D.prepAxes()
backendCluster.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b')
mapCluster.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r')
ax.set_title('after ICP')

plt.show()

