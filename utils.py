from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gtsam import Pose2

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

def plot_pose2_on_axes(axes,
                       pose: Pose2,
                       axis_length: float = 0.1,
                       covariance: np.ndarray = None) -> None:
    """
    TAKEN FROM gtsam.utils.plot AND SLIGHTLY EDITED


    Plot a 2D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        pose: The pose to be plotted.
        axis_length: The length of the camera axes.
        covariance (numpy.ndarray): Marginal covariance matrix to plot
            the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    t = pose.translation()
    origin = t

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    graphics_line1 = axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    graphics_line2 = axes.plot(line[:, 0], line[:, 1], 'g-')

    if covariance is not None:
        pPp = covariance[0:2, 0:2]
        gPp = np.matmul(np.matmul(gRp, pPp), gRp.T)

        w, v = np.linalg.eig(gPp)

        # k = 2.296
        k = 5.0

        angle = np.arctan2(v[1, 0], v[0, 0])
        e1 = Ellipse(origin,
                np.sqrt(w[0] * k),
                np.sqrt(w[1] * k),
                np.rad2deg(angle),
                fill=False)
        axes.add_patch(e1)

    return graphics_line1[0], graphics_line2[0], e1

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