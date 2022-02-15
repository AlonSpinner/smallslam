#from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gtsam import Pose2

import map
import robot

def plot_cov_ellipse(pos, cov, nstd=1, ax=None, facecolor = 'none',edgecolor = 'b' ,  **kwargs):
        #slightly edited from https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        '''
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.

        Parameters
        ----------
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            cov : The 2x2 covariance matrix to base the ellipse on
            nstd : The radius of the ellipse in numbers of standard deviations.
            ax : The axis that the ellipse will be plotted on. If not provided, we won't plot.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        '''
        eigs, vecs = np.linalg.eig(cov)
        theta = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #obtain theta from first axis. second axis is just perpendicular to it

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(eigs)
        ellip = Ellipse(xy=pos, 
                        width=width, 
                        height=height, 
                        angle=theta,
                        facecolor = facecolor, 
                        edgecolor=edgecolor, **kwargs)

        if ax is not None:
            ax.add_patch(ellip)
        
        return ellip

def plot_pose(axes , Rp2g, origin, axis_length: float = 0.1, covariance: np.ndarray = None):
    '''
    TAKEN FROM gtsam.utils.plot AND SLIGHTLY EDITED


    Plot a 2D pose on given axis `axes` with given `axis_length`.
    '''

    x_axis = origin + Rp2g[:, 0] * axis_length
    line = np.vstack((origin,x_axis))
    graphics_line1, = axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + Rp2g[:, 1] * axis_length
    line = np.vstack((origin,y_axis))
    graphics_line2, = axes.plot(line[:, 0], line[:, 1], 'g-')


    if covariance is not None:
        graphics_ellip = plot_cov_ellipse(origin, covariance[:2,:2], nstd=1, ax=axes, facecolor = 'none', edgecolor = 'k')

        
   

    return graphics_line1, graphics_line2, graphics_ellip

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