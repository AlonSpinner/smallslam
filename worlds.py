import numpy as np
import matplotlib.pyplot as plt
import map
import robot


def setWorldMap():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-2,2); ax.set_ylim(-1,3); 
    ax.set_xlabel('x'); ax.set_ylabel('y'); 
    ax.set_aspect('equal'); ax.grid()
    return fig, ax

def default_world():
    np.random.seed(seed=2) #this is important. affects measurements aswell

    #------Build worldmap
    fig , ax = setWorldMap()
    N = 40 #number of landmarks
    semantics = ("table","MEP","chair","pillar")
    xrange = (-2,2)
    yrange = (-1,3)

    landmarks = [None] * N
    for ii in range(N):
        landmarks[ii] = map.landmark(x = (np.random.rand()-0.5) * np.diff(xrange) + np.mean(xrange),
                                    y = (np.random.rand()-0.5) * np.diff(yrange) + np.mean(yrange),
                                    classLabel = np.random.choice(semantics),
                                )

    worldMap = map.map(landmarks)
    worldMap.plot(ax = ax, plotIndex = True,plotCov = False)

    #------Spawn Robot
    car = robot.robot(ax = ax)
    car.range = 2
    
    return car, worldMap, ax, fig

def default_parameters():
    #dictionary form
    
    #build sub dictionary odom
    odom = (0.4,0.4,0.4) #dx,dy,dtheta

    #build sub dictionary time
    time = {
            "dt": 0.5,
            "timeFinal": 10,
            }

    #build prms
    prms = {
            "odom": odom,
            "time": time,
            }
    return prms