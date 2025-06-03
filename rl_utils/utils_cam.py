import torch
from torch import nn
import numpy as np
from rl_utils.utils_graphic import getWorld2View2, getProjectionMatrix
# from utils.general_utils import PILtoTorch
import cv2

        
class MiniCam:
    def __init__(self, R, T, width, height, fovy, fovx, znear=0.01, zfar=100.0):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform =  torch.from_numpy(getWorld2View2(R, T, np.array([0,0,0]), 1.0).astype(np.float32)).transpose(0, 1).contiguous().cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).contiguous().cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera(nn.Module):
    def __init__(self, R, T, W, H,FoVx, FoVy,
                 image_name=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = W
        self.image_height = H

        self.depth_reliable = False
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def getWorld2View3(R, t, translate=np.zeros(3), scale=1.0):
    # camera to world 
    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, :3] = R
    C2W[:3, 3] = (t + translate) * scale

    # world to camera
    W2C = np.linalg.inv(C2W).astype(np.float32)
    return W2C

if __name__ == "__main__":
    pose_data = np.array([
    -3.2056962200324568e-01,  4.4805519469580130e-01, -8.3455476749869673e-01,  3.4529874164067347e+00,
     9.4722495609474755e-01,  1.5163545203918455e-01, -2.8243861675800630e-01,  4.5461101341359472e-01,
     1.0789779324904232e-16, -8.8105234361584872e-01, -4.7301878166624667e-01,  5.9362854471594151e-01,
     0.0,                     0.0,                     0.0,                     1.0
     ])
    pose_mat = pose_data.reshape((4,4)) # [R,t]
    R = pose_mat[:3, :3]
    T = pose_mat[:3,  3]
    
    width, height = 1200, 680
    fx, fy = 600.0, 600.0
    fovx = 2 * np.arctan(width  / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    
    cam = MiniCam(
        R=R,
        T=T,
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=0.01,
        zfar=100.0
    )
    print(cam.world_view_transform)
    print("--"*30)
    print(cam.projection_matrix)
    print("--"*30)
    print(cam.full_proj_transform)
    print("--"*30)
    print(cam.camera_center)
    print("--"*30)