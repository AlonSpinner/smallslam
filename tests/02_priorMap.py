import numpy as np
import smallslam.map as map
import matplotlib.pyplot as plt

np.random.seed(seed=2)

classLabels = ("table","MEP","chair","pillar","clutter")
xrange = (-2,2)
yrange = (-1,3)
sigmarange = (-0.5,0.5)

priorMap = map.map()
priorMap.fillMapRandomly(20,classLabels, xrange, yrange, sigmarange)
priorMap.plot(plotIndex = True,plotCov = True, plotLegend = True ,markerSize = 30)
plt.show()