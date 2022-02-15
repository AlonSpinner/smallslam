import numpy as np
import map
import robot
import utils

def world1():
    np.random.seed(seed=2) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-2,2); yrange = (-1,3)
    fig , ax = utils.spawnWorld(xrange, yrange)
    
    #------landmarks
    N = 40 
    semantics = ("table","MEP","chair","pillar")
    worldMap = map.map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange)
    worldMap.plot(ax = ax, plotIndex = True,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = ax)
    car.range = 2
    
    #------ ground truth odometrey
    dx = 0.2; dy = 0.2; dtheta =0.2
    odom = [np.array([dx,dy,dtheta])] * 50

    return car, worldMap, ax, fig, odom