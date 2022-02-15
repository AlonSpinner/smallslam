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
    dx = 0.4; dy = 0.4; dtheta =0.4
    odom = [[dx,dy,dtheta]] * 30

    return car, worldMap, ax, fig, odom