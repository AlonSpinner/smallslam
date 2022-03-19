import numpy as np
import matplotlib.pyplot as plt
from smallslam.utils.datatypes import landmark_mesh
from smallslam.map import map

p1 = np.array([1,0]); cov1 = np.array([[0.1,0],[0,0.1]])
p2 = np.array([1,1]); cov2 = np.array([[0.2,0],[0,0.1]])
p3 = np.array([0,1]); cov3 = np.array([[0.1,0],[0,0.2]])
p4 = np.array([0,2]); cov4 = np.array([[0.1,0],[0,0.2]])

mesh1 = landmark_mesh(id = 0, 
                      u = np.array([0,1]),
                      xy = np.array([p1,p2]),
                      cov = np.array([cov1,cov2]),
                      classLabel = 'wall')

mesh2 = landmark_mesh(id = 1, 
                      u = np.array([0,1]),
                      xy = np.array([p2,p3]),
                      cov = np.array([cov2,cov3]),
                      classLabel = 'wall')

mesh3 = landmark_mesh(id = 2, 
                      u = np.array([0,1]),
                      xy = np.array([p3,p4]),
                      cov = np.array([cov3,cov4]),
                      classLabel = 'wall')

intrpMap = map()
intrpMap.addLandmarks(mesh1.interpolate(5))
intrpMap.addLandmarks(mesh2.interpolate(8))
intrpMap.addLandmarks(mesh3.interpolate(20))
ax = intrpMap.plot(plotCov = True, plotLegend = True, plotMeshIndex = True)
ax.set_title('interp map')

dataMap = map()
dataMap.addLandmarks(mesh1.export())
dataMap.addLandmarks(mesh2.export())
dataMap.addLandmarks(mesh3.export())
ax = dataMap.plot(plotCov = True, plotLegend = True, plotMeshIndex = True)
ax.set_title('data map')

plt.show()
