import numpy as np

class CentralCamera:
    def __init__(self, focal=0.015, pixel=10e-6, 
                 resolution=(1280, 1024), center=(640, 512)):
        """
        :param focal: Focal length (meters) [default: 15mm]
        :param pixel: Pixel size (meters) [default: 10um]
        :param resolution: (width, height) pixels
        :param center: (u0, v0) principal point
        """
        self.f = focal
        self.rho = pixel
        self.w, self.h = resolution
        self.u0, self.v0 = center
        
        # Compute Intrinsic Matrix K (in pixels)
        # fx = fy = f / rho
        fx = fy = self.f / self.rho
        
        self.K = np.array([[fx,  0, self.u0],
                           [ 0, fy, self.v0],
                           [ 0,  0,       1]])

    def project(self, P_world):
        """Projects a 3D point (X,Y,Z) to 2D pixels (u,v)"""
        # Linear Projection
        p_hom = self.K @ P_world
        
        # Perspective Division
        uv = p_hom[:2] / p_hom[2]
        return uv

