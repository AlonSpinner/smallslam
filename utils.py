import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def setWorldMap():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-2,2); ax.set_ylim(-1,3); 
    ax.set_xlabel('x'); ax.set_ylabel('y'); 
    ax.set_aspect('equal'); ax.grid()
    return fig, ax

def plot_cov_ellipse(cov, pos, nstd=1, ax=None, facecolor = 'none', **kwargs):
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
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor = facecolor, **kwargs)

        ax.add_artist(ellip)
        return ellip