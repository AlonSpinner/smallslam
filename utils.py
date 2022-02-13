from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import map
import robot

def plot_cov_ellipse(cov, pos, nstd=1, ax=None, facecolor = 'none',edgecolor = 'b' ,  **kwargs):
        #slightly edited from https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        '''
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        '''
        def eigsorted(cov):
            eigs, vecs = np.linalg.eigh(cov)
            order = eigs.argsort()[::-1]
            if np.any(eigs < 0):
                raise TypeError("covaraince matrix must be positive definite")
            return eigs[order], vecs[:,order]

        if ax is None:
            ax = plt.gca()

        eigs, vecs = eigsorted(cov) #I am not exactly sure why this is needed
        theta = np.degrees(np.arctan2(vecs[1,0],vecs[0,0])) #obtain theta from first axis. second axis is just perpendicular to it

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(eigs)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, \
         facecolor = facecolor, edgecolor=edgecolor, **kwargs)

        ax.add_artist(ellip)
        return ellip

def setWorldMap():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-2,2); ax.set_ylim(-1,3); 
    ax.set_xlabel('x'); ax.set_ylabel('y'); 
    ax.set_aspect('equal'); ax.grid()
    return fig, ax

def default_world():
    np.random.seed(seed=2) #this is important. affects measuremetns aswell

    #------Build worldmap
    fig , ax = setWorldMap()
    N = 15 #number of landmarks
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