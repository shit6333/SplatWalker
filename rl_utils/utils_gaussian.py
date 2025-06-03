import os
import torch
from torch import nn
from einops import rearrange
import numpy as np
from plyfile import PlyData, PlyElement
from os import makedirs, path
from errno import EEXIST

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

C0 = 0.28209479177387814

# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions,
    eps=1e-8,
) :
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale,
    rotation_xyzw,
):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class GaussianModel:

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # self._semantic_feature = torch.empty(0) 
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        # self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    # def get_semantic_feature(self):
    #     return self._semantic_feature

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        # for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):  
            # l.append('semantic_{}'.format(i))
        return l
    
    def to_cuda(self, device: str = 'cuda'):
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                # print(f"Moving {name}: {val.shape} from {val.device} to {device}")
                setattr(self, name, val.to(device))
        return self
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self._opacity).detach().cpu().numpy()
        scale = torch.log(self._scaling).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) 
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        """
        ply => torch tensor
        """
        plydata = PlyData.read(path)
        element = plydata.elements[0]

        # load xyz and opacity
        xyz = np.stack((
            np.asarray(element["x"]),
            np.asarray(element["y"]),
            np.asarray(element["z"])
        ), axis=1)
        opacities = np.asarray(element["opacity"])[..., np.newaxis]

        # read DC feature (e.g f_dc_0, f_dc_1, f_dc_2)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(element["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(element["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(element["f_dc_2"])

        # sh feature (f_rest_*)
        extra_f_names = [p.name for p in element.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        expected = 3 * (self.max_sh_degree + 1) ** 2 - 3
        assert len(extra_f_names) == expected, f"Expect {expected} f_rest_，but got {len(extra_f_names)}"
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(element[attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scale
        scale_names = [p.name for p in element.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(element[attr_name])

        # rotation 資料
        rot_names = [p.name for p in element.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(element[attr_name])

        # to torch tensor
        self._xyz = torch.tensor(xyz, dtype=torch.float)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float)
        self._scaling = torch.tensor(scales, dtype=torch.float)
        self._rotation = torch.tensor(rots, dtype=torch.float)

        self.active_sh_degree = self.max_sh_degree
        print(f"Loaded ply file: {path}")


if __name__ == '__main__':
    pretrain_ply_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room/params.ply'
    pretrain_ply_path = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/fe421606-c/point_cloud/iteration_7000/point_cloud.ply'
    
    sh_degree = 0
    gaussian = GaussianModel(sh_degree=sh_degree)
    gaussian.load_ply(pretrain_ply_path)

    print(gaussian.get_features.shape)
    print(gaussian._features_dc.shape)
    print(gaussian._features_rest.shape)
    print(gaussian.get_opacity.shape)
    print(gaussian.get_xyz.shape)
    print(gaussian.get_scaling.shape)
    print(gaussian.get_rotation.shape)