from dataclasses import dataclass
import numpy as np
import gtsam

@dataclass(frozen = True, order = True)
class landmark:
    id : int
    xy : np.ndarray((2))
    cov : np.ndarray((2,2))
    classLabel : str
    mesh_id : int = 0 #0 - belongs to no known mesh


@dataclass(frozen = True)
class meas_odom:
    dpose : gtsam.Pose2
    cov : np.ndarray((2,2))

@dataclass(frozen = True, order = True)
class meas_landmark:
        id : int
        range : float
        angle : float
        cov : np.ndarray((2,2))
        classLabel : str

@dataclass(frozen = True, order = True)
class landmark_mesh:
    id: int
    u : np.ndarray # (m) arclength parameter spanning [0,1]
    xy : np.ndarray #(m,1,2)
    cov : np.ndarray #(m,2,2)
    classLabel : str

    def export(self):
        landmarks = [None] * len(self.u)
        for ii in range(len(self.u)):
              landmarks[ii] = landmark(id = ii,
                                     xy = self.xy[ii], 
                                     cov = self.cov[ii],
                                     classLabel = self.classLabel,
                                     mesh_id = self.id,
                                     )
        return landmarks

    def interpolate(self,N):
        q = np.linspace(0,1,N)
        
        xq = np.interp(q,self.u,self.xy[:,0])
        yq = np.interp(q,self.u,self.xy[:,1])
        xyq = np.vstack((xq,yq)).T

        covq = np.zeros((N,2,2))
        covq[:,0,0] = np.interp(q,self.u,self.cov[:,0,0])
        covq[:,0,1] = np.interp(q,self.u,self.cov[:,0,1])
        covq[:,1,0] = covq[:,0,1] #symmetry. if bigger than 2D should have also transposed
        covq[:,1,1] = np.interp(q,self.u,self.cov[:,1,1])
        
        landmarks = [None] * N
        for ii in range(N):
              landmarks[ii] = landmark(id = ii,
                                     xy = xyq[ii], 
                                     cov = covq[ii],
                                     classLabel = self.classLabel,
                                     mesh_id = self.id,
                                     )
        return landmarks

def pose2ToNumpy(pose2: gtsam.Pose2):
    return np.array([pose2.x(),pose2.y(),pose2.theta()])
