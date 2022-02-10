import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np

class map:

    #class attributes
    markersList = ["o","v","s","D","4","*",">","<","2","P"] #10 classes max
    
    def __init__(self,landmarks = []):
        '''
        each landmark is dictionary:
        {x - float,
        y - float,
        classLabel - string,
        covaraince - 2x2 float matrix,
        index - integer}. 
        '''

        #instance attributes
        self.landmarks = []
        self.addLandmarks(landmarks)
        
        #wannabe class attribute 
        self.colors = self.randomColors('viridis',len(self.markersList))

    def addLandmarks(self,landmarks):
        self.landmarks.extend(landmarks)
        self.defineSemanticsFromLandmarks()
        self.indexifyLandmarks()

    #goes over all landmarks to find classLabels. semantics is a list of unique classLabels
    def defineSemanticsFromLandmarks(self):
        if self.landmarks: #not empty
            classLabels = [lm['classLabel'] for lm in self.landmarks]
            semantics = list(set(classLabels))
            if len(semantics) > 10:
                raise TypeError("no more than 10 classes are premited. Not enough distinguishable markers in matplotlib")
            self.semantics = list(set(classLabels))

    def indexifyLandmarks(self):
        for ii in range(len(self.landmarks)):
                self.landmarks[ii]["index"] = ii

    def plot(self,ax = None, plotIndex = False, plotCov = True):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()

        for lm in self.landmarks:
            ii = self.semantics.index(lm["classLabel"]) #index of classLabel in semantics. used for shape and color
            ax.scatter(lm["x"],lm["y"],
                marker = self.markersList[ii],
                c = self.colors[ii].reshape(1,-1) #reshape to prevent warnning. scatter wants 2D array
                )
            if plotIndex:    
                ax.text(lm["x"],lm["y"],lm["index"])

            if plotCov:
                self.plot_cov_ellipse(lm["cov"],(lm["x"],lm["y"]),nstd = 1,ax = ax,edgecolor=self.colors[ii].reshape(1,-1))


    #define color for each classLabel.
    @staticmethod 
    def randomColors(mapname,N):
        np.random.seed(seed=0) #ensure colors-labels are the same for each map 
        return np.random.permutation(plt.cm.get_cmap(mapname,N).colors) #permuate to make colors more distinguishable if not alot of classes are used

    @staticmethod
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



    

    
