from smallslam.map import map
from smallslam.utils.datatypes import landmark_mesh
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

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
    observeableMap.landmarks = lms[::2]
    observeableMap.fillMapRandomly(N = 20, classLabels = ['clutter'], xrange = (0.0,4.0), yrange=(0.0,3.0), sigmarange = (0.001,0.1))
    
    if plot:
        ax = modelMap.plot(plotCov = True, plotLegend = True, plotMeshIndex = True)
        ax.set_title('Model map')    
        
        ax = observeableMap.plot(plotCov = False, plotLegend = True, plotMeshIndex = True)
        ax.set_title('observable map')
        
        plt.show()

    return modelMap, observeableMap


buildMaps(plot = True)