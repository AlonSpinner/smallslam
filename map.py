import matplotlib.pyplot as plt
import numpy as np
import utils

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
            classLabels = [lm.classLabel for lm in self.landmarks]
            semantics = list(set(classLabels))
            if len(semantics) > 10:
                raise TypeError("no more than 10 classes are premited. Not enough distinguishable markers in matplotlib")
            self.semantics = list(set(classLabels))

    def indexifyLandmarks(self):
        for ii in range(len(self.landmarks)):
                self.landmarks[ii].index = ii

    def plot(self,ax = None, plotIndex = False, plotCov = False):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()

        for lm in self.landmarks:
            ii = self.semantics.index(lm.classLabel) #index of classLabel in semantics. used for shape and color
            ax.scatter(lm.x,lm.y,
                marker = self.markersList[ii],
                c = self.colors[ii].reshape(1,-1) #reshape to prevent warnning. scatter wants 2D array
                )
            if plotIndex:    
                ax.text(lm.x,lm.y,lm.index)

            if plotCov:
                if not len(lm.cov): #if is empty
                    raise TypeError("landmark has no covaraince key")
                utils.plot_cov_ellipse(lm.cov,(lm.x,lm.y),nstd = 1,ax = ax,edgecolor=self.colors[ii].reshape(1,-1))

    #define color for each classLabel.
    @staticmethod 
    def randomColors(mapname,N):
        np.random.seed(seed=0) #ensure colors-labels are the same for each map 
        return np.random.permutation(plt.cm.get_cmap(mapname,N).colors) #permuate to make colors more distinguishable if not alot of classes are used    

class landmark: #need to work on this later... switch landmark dictionary representation to class
    def __init__(self,x = 0, y = 0, classLabel = 'clutter', cov = [], index = []):
        self.x = x
        self.y = y
        self.classLabel = classLabel
        self.cov = cov
        self.index = index
        return


    

    
