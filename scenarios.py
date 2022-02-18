import numpy as np
import map
import robot
import utils
import gtsam

def scenario1():
    np.random.seed(seed=2) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-2,2); yrange = (-2,2)
    fig , ax = utils.spawnWorld(xrange, yrange)
    
    #------landmarks
    N = 40 
    semantics = ("table","MEP","chair","pillar")
    worldMap = map.map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange)
    worldMap.plot(ax = ax, plotIndex = True,plotCov = False)

    #------Spawn Robot
    pose0 = gtsam.Pose2(1.0,0.0,np.pi/2)
    car = robot.robot(ax = ax, pose = pose0, FOV = np.radians(90), range = 2)
    
    #------ ground truth odometrey
    r = 1; m = 20
    dx = 0.309017 ;dy = 0.0489435; dtheta = 0.314159
    gt_odom = [gtsam.Pose2(dx,dy,dtheta)] * 2*m

    return car, worldMap, ax, fig, gt_odom


def scenario2():
    np.random.seed(seed=3) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-10,+100); yrange = (-5,+5)
    fig , (axWorld,axError) = utils.spawnWorld(xrange, yrange, type = "world and error")
    
    #------landmarks
    N = 100 
    semantics = ("table","MEP","chair")
    worldMap = map.map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange = (-2,+2))
    worldMap.plot(ax = axWorld, plotIndex = False,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = axWorld, FOV = np.radians(90), markerSize = 50)
    car.range = 10
    
    #------ ground truth odometrey
    dx = 1; dy = 0; dtheta =0
    odom = [np.array([dx,dy,dtheta])] * 100

    return car, worldMap, axWorld, axError,  fig, odom

def scenario3():
    np.random.seed(seed=3) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-10,+100); yrange = (-5,+30)
    fig , (axWorld,axError) = utils.spawnWorld(xrange, yrange, type = "world and error")
    
    #------landmarks
    N = 100 
    semantics = ("table","MEP","chair")
    worldMap = map.map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange = (-2,+2))
    worldMap.plot(ax = axWorld, plotIndex = False,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = axWorld, FOV = np.radians(90), markerSize = 50)
    car.range = 3
    
    #------ ground truth odometrey
    dx = 1; dy = 0; dtheta =0
    odom = [np.array([dx,dy,dtheta])] * 100

    return car, worldMap, axWorld, axError,  fig, odom

def scenario4():
    np.random.seed(seed=3) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-10,+100); yrange = (-5,+30)
    fig , (axWorld,axError) = utils.spawnWorld(xrange, yrange, type = "world and error")
    
    #------landmarks
    N = 100 
    semantics = ("table","MEP","chair")
    worldMap = map.map()
    worldMap.fillMapRandomly(N,semantics,xrange,yrange = (-10,+10))
    worldMap.plot(ax = axWorld, plotIndex = False,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = axWorld, FOV = np.radians(90), markerSize = 50)
    car.range = 20
    
    #------ ground truth odometrey
    dx = 1; dy = 0; dtheta =0
    odom = [np.array([dx,dy,dtheta])] * 100

    return car, worldMap, axWorld, axError, fig, odom

def scenario5():
    np.random.seed(seed=3) #this is important. affects measurements aswell

    #------Build worldmap
    xrange = (-5,+25); yrange = (-5,+25)
    fig , (axWorld,axError) = utils.spawnWorld(xrange, yrange, type = "world and error")
    
    #------landmarks
    N = 100 
    semantics = ("table","MEP","chair")
    worldMap = map.map()
    #worldMap.fillMapRandomly(N,semantics,xrange,yrange)
    worldMap.plot(ax = axWorld, plotIndex = False,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = axWorld, FOV = np.radians(90), markerSize = 50)
    car.range = 10
    
    #------ ground truth odometrey
    a = 20 #square side length
    odomLine = [np.array([a/20,0,0])] * 20
    odomTurn = [np.array([0,0,np.pi/2/5])] * 5
    
    odom = []
    for _ in range(4):
        odom = odom + odomLine + odomTurn 

    return car, worldMap, axWorld, axError, fig, odom