import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Dict
from collections import deque

from gaussian_renderer import render
from rl_utils.utils_cam import MiniCam
from rl_utils.utils_pipe import DummyPipeline
from rl_utils.utils_gaussian import GaussianModel
from rl_utils.utils_colmap import read_images_binary 
from aesthetics_model import AestheticsModel
from PIL import Image

def resize_numpy(img, s=(240,240)):
    img_torch = torch.from_numpy(img).unsqueeze(0)
    img = F.interpolate(img_torch, size=s, mode='bilinear', align_corners=False)
    img = img.squeeze(0).numpy()
    return img

def random_camera_pose_normal(gaussian_xyz: torch.Tensor,
                              yaw_range: float = np.pi,
                              pitch: float = 0.0,
                              init_var_scale: float = 0.8,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    xyz = gaussian_xyz.detach().cpu().numpy()
    mean_xyz = xyz.mean(axis=0)
    var_xyz = xyz.var(axis=0) * init_var_scale
    T = np.random.normal(loc=mean_xyz, scale=np.sqrt(var_xyz)).astype(np.float32)
    yaw = np.random.uniform(-yaw_range, yaw_range)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32)
    R_y = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    R = R_y @ R_x
    return R, T



def random_camera_pose_sample(init_poses):
    H = random.choice(init_poses)
    H_inv = np.linalg.inv(H)
    R = H[:3,:3]
    T = H_inv[:3, 3]
    return R, T

def selected_camera_pose_sample(init_poses, idx):
    H = init_poses[idx]
    H_inv = np.linalg.inv(H)
    R = H[:3,:3]
    T = H_inv[:3, 3]
    return R, T

def compute_variance_penalty(
    images: torch.Tensor,
    sigma_ref: float = 0.2,
    lambda_penalty: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert RGB batch to grayscale by averaging channels
    gray = images.mean(dim=1)  # [B, H, W]
    flat = gray.view(gray.size(0), -1)
    std = torch.std(flat, dim=1, unbiased=False)
    penalty = lambda_penalty * torch.clamp(1.0 - std / sigma_ref, min=0.0)
    return std, penalty

def compute_uniformity_penalty(
    images: torch.Tensor,
    bins: int = 16,
    lambda_penalty: float = 1.0
) -> torch.Tensor:
    gray = images.mean(dim=1)
    B = gray.size(0)
    penalties = []
    for i in range(B):
        flat = gray[i].view(-1)
        hist = torch.histc(flat, bins=bins, min=0.0, max=1.0)
        p = hist / hist.sum()
        uniform = torch.full_like(p, 1.0 / bins)
        tvd = 0.5 * torch.sum(torch.abs(p - uniform))
        penalties.append(lambda_penalty * tvd)
    return torch.stack(penalties)

class UnifiedGaussianEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
        ply_path: str,
        *,
        aes_model: AestheticsModel,
        img_width: int = 1200,
        img_height: int = 680,
        fx: float = 600.0,
        fy: float = 600.0,
        images_bin = None,
        sh_degree: int = 0,
        max_steps: int = 15, # 20
        init_var_scale: float = 0.6,
        shift_scale: float = 0.1,
        rotate_scale: float = 0.05,
        diversity_radius: float = 1.0,
        lambda_avg: float = 0.3,
        smooth_window: int = 3,
        history_length: int = 3,
        excluding_length: int =3,
        device: str = "cuda",
        z_rotate: bool = False
    ):
        super().__init__()
        # basic sim
        self.device = device
        self.gaussian = GaussianModel(sh_degree=sh_degree)
        self.gaussian.load_ply(ply_path)
        self.gaussian.to_cuda()
        self.pipeline = DummyPipeline()
        self.background = torch.zeros(3, device=device)
        self.W, self.H, self.fx, self.fy = img_width, img_height, fx, fy
        self.fovx = 2*np.arctan(img_width/(2*fx))
        self.fovy = 2*np.arctan(img_height/(2*fy))
        self.shift_scale, self.rotate_scale = shift_scale, rotate_scale
        self.max_steps = max_steps
        self.init_var_scale = init_var_scale
        self.images_bin = images_bin
        self.init_poses = None
        if self.images_bin is not None:
            self.init_poses, _ = read_images_binary(self.images_bin)
        

        # reward machinery
        self.aes_model = aes_model
        self.diversity_radius = diversity_radius
        self.smooth_window = smooth_window
        self.history_length = history_length
        self.excluding_length = excluding_length

        self.past_actions = deque(maxlen=smooth_window)
        self.pose_history = deque(maxlen=history_length)
        self.excluding_seqs = deque(maxlen=excluding_length)
        self.aes_history = []
        self.lambda_avg = lambda_avg
        self.z_rotate = z_rotate

        # spaces
        # self.action_space = spaces.Box(-1.0,1.0,shape=(5,),dtype=np.float32)
        if self.z_rotate:
            self.action_space = spaces.Box(-1.0,1.0,shape=(6,),dtype=np.float32)
        else:
           self.action_space = spaces.Box(-1.0,1.0,shape=(5,),dtype=np.float32) 
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0.0,1.0,shape=(3, 84, 84),dtype=np.float32),
            "pose": spaces.Box(-np.inf,np.inf,shape=(6,),dtype=np.float32),
            "history_poses": spaces.Box(-np.inf,np.inf,shape=(history_length,6),dtype=np.float32),
            "excluding_poses": spaces.Box(-np.inf,np.inf,shape=(excluding_length,6),dtype=np.float32),
        })
        # self.observation_space = spaces.Dict({
        #     "image": spaces.Box(0.0,1.0,shape=(3,self.H,self.W),dtype=np.float32),
        #     "pose": spaces.Box(-np.inf,np.inf,shape=(6,),dtype=np.float32),
        #     "history_poses": spaces.Box(-np.inf,np.inf,shape=(history_length,6),dtype=np.float32),
        #     "excluding_poses": spaces.Box(-np.inf,np.inf,shape=(excluding_length,6),dtype=np.float32),
        # })

        # init state
        self.R = np.eye(3,dtype=np.float32)
        self.T = np.zeros(3,dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None, idx=None):
        self.step_count = 0
        self.aes_history = []
        self.past_actions.clear()
        self.pose_history.clear()
        # self.excluding_seqs.clear()

        if self.images_bin is not None:
            self.R, self.T = random_camera_pose_sample(self.init_poses)
            img = self._render()
        else:
            for _ in range(20):
                self.R, self.T = random_camera_pose_normal(self.gaussian.get_xyz, init_var_scale=self.init_var_scale)
                img = self._render()
                black_mask = (img < 1e-4).all(axis=0)   # shape (H,W) bool
                black_ratio = black_mask.mean()         # float
                if black_ratio <= 0.30:
                    break   

        if idx is not None:
            self.R, self.T = selected_camera_pose_sample(self.init_poses, idx)
            img = self._render()

        pose = self._get_pose()
        img = resize_numpy(img, (84,84))
        return self._pack_obs(img, pose), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        # translate
        dx,dy,dz = action[:3]*self.shift_scale
        self.T += np.array([dx,dy,dz],dtype=np.float32)
        # rotate
        if self.z_rotate:
            dyaw, dpitch, droll = action[3:] * self.rotate_scale
            Rx = np.array([[1,0,0],[0,np.cos(dpitch),-np.sin(dpitch)],[0,np.sin(dpitch),np.cos(dpitch)]],dtype=np.float32)
            Ry = np.array([[np.cos(dyaw),0,np.sin(dyaw)],[0,1,0],[-np.sin(dyaw),0,np.cos(dyaw)]],dtype=np.float32)
            Rz = np.array([[ np.cos(droll),-np.sin(droll),0],[np.sin(droll),np.cos(droll),0],[0,0,1]],dtype=np.float32)
            self.R = Rz @ Ry @ Rx @ self.R
        else:
            dyaw,dpitch = action[3:]*self.rotate_scale
            Rx = np.array([[1,0,0],[0,np.cos(dpitch),-np.sin(dpitch)],[0,np.sin(dpitch),np.cos(dpitch)]],dtype=np.float32)
            Ry = np.array([[np.cos(dyaw),0,np.sin(dyaw)],[0,1,0],[-np.sin(dyaw),0,np.cos(dyaw)]],dtype=np.float32)
            self.R = Ry @ Rx @ self.R

        img = self._render()
        pose = self._get_pose()
        # print(img.shape)
        # aesthetic
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img_t = F.interpolate(img_t, size=(240,240), mode='bilinear', align_corners=False)
        aes_score = float(self.aes_model(img_t))
        self.aes_history.append(aes_score)
        std, penalty = compute_variance_penalty(img_t, sigma_ref=0.2, lambda_penalty=3.0)
        # penalty = compute_uniformity_penalty(img_t, lambda_penalty=1.2)
        penalty = float(penalty[0].item())
        # penalty = 0.0

        # diversity
        if self.excluding_seqs:
            dists = [np.linalg.norm(pose[:3]-ex[:3]) for ex in self.excluding_seqs]
            div_ratio = min(min(dists)/self.diversity_radius,1.0)
        else:
            div_ratio = 1.0
            
        # smoothness
        if self.past_actions:
            avg = np.mean(self.past_actions,axis=0)
            diff = np.linalg.norm(action-avg)
            c = max(np.linalg.norm(avg)/2,0.1)
            smo_ratio = np.exp(-diff*diff/(2*c*c))
        else:
            smo_ratio = 1.0
        
        smo_ratio = 1.0
        div_ratio = 1.0
        penalty = 0.0
        reward = aes_score * div_ratio * smo_ratio - penalty

        self.past_actions.append(action.copy())
        self.pose_history.append(pose.copy())
        if self.step_count>=self.max_steps:
            done=True
            self.excluding_seqs.append(pose.copy())
        else:
            done=False

        # done reward bonus
        end_bonus = 0
        if done:
            avg_aes = sum(self.aes_history) / len(self.aes_history)
            end_bonus = self.lambda_avg * avg_aes
            # reward += end_bonus

        img = resize_numpy(img, (84,84))
        obs = self._pack_obs(img,pose)
        info = {"aesthetic":aes_score, "diversity":div_ratio, "smoothness":smo_ratio, "penalty":penalty, "end_bouns":end_bonus}
        return obs, reward, done, False, info

    def _render(self):
        # print(self.R)
        cam = MiniCam(R=self.R, T=self.T,
                     width=self.W, height=self.H,
                     fovx=self.fovx, fovy=self.fovy,
                     znear=0.01, zfar=100.0)
        return render(cam,self.gaussian,self.pipeline,self.background)["render"].detach().cpu().numpy()
    
    def render(self):
        return (torch.from_numpy(self._render()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def _render_by_pose(self, R, T):
        cam = MiniCam(R=R, T=T,
                     width=self.W, height=self.H,
                     fovx=self.fovx, fovy=self.fovy,
                     znear=0.01, zfar=100.0)
        return render(cam,self.gaussian,self.pipeline,self.background)["render"].detach().cpu().numpy()

    def _get_pose(self):
        yaw = np.arctan2(self.R[0,2],self.R[2,2])
        pitch = np.arcsin(np.clip(-self.R[1,2],-1,1))
        roll = 0.0
        if self.z_rotate:
            roll = np.arctan2(self.R[1,0], self.R[1,1])
        return np.concatenate([self.T.astype(np.float32),
                               np.array([yaw,pitch,roll],dtype=np.float32)],axis=0)

    def _pack_obs(self, img, pose):
        # history
        hist = list(self.pose_history)
        pad = [np.zeros(6,dtype=np.float32)]*(self.history_length-len(hist))
        history = np.stack(pad+hist[-self.history_length:],axis=0)
        # history pose
        hist = list(self.past_actions)
        pad = [np.zeros(5,dtype=np.float32)]*(self.smooth_window-len(hist))
        history_action = np.stack(pad+hist[-self.smooth_window:],axis=0)
        # excluding
        seq = list(self.excluding_seqs)
        pad_n = self.excluding_length-len(seq)
        pad = [np.random.uniform(-1,1,6).astype(np.float32) for _ in range(pad_n)]
        excluding = np.stack(pad+seq[-self.excluding_length:],axis=0)
        return {
            "image": img,
            "pose": pose,
            "history_poses": history,
            "history_actions": history_action,
            "excluding_poses": excluding,
        }

if __name__=="__main__":
    # ... argparse same as before ...
    pass

    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Test UnifiedGaussianEnv")
    parser.add_argument("--ply",type=str,default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/garden/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--steps", type=int, default=20, help="max steps per episode")
    parser.add_argument("--hist_len", type=int, default=5, help="history length")
    parser.add_argument("--excl_len", type=int, default=3, help="excluding length")
    parser.add_argument("--div_rad", type=float, default=1.0, help="diversity radius")
    parser.add_argument("--smooth_win", type=int, default=3, help="smoothness window")
    parser.add_argument( "--w", type=int, default=1557, help="image width")
    parser.add_argument("--h", type=int, default=1038, help="image height")
    parser.add_argument("--fx", type=float, default=1586.0, help="focal x")
    parser.add_argument("--fy", type=float, default=1586.0, help="focal y")
    parser.add_argument("--init_var_scale", type=float, default=0.6, help="initial pose variance scale")
    parser.add_argument("--device", type=str, default="cuda", help="torch device")
    args = parser.parse_args()

    # aesthetic model
    aes_model = AestheticsModel(device=args.device)

    # build env
    env = UnifiedGaussianEnv(
        ply_path=args.ply,
        aes_model=aes_model,
        img_width=args.w,
        img_height=args.h,
        fx=args.fx,
        fy=args.fy,
        max_steps=args.steps,
        init_var_scale=args.init_var_scale,
        diversity_radius=args.div_rad,
        smooth_window=args.smooth_win,
        history_length=args.hist_len,
        excluding_length=args.excl_len,
        device=args.device,
        sh_degree=3,
        images_bin = '/mnt/HDD3/miayan/omega/RL/gaussian-splatting/data/garden/sparse/0/images.bin'
    )

    obs, info = env.reset()
    obs_render = env.render()
    img = Image.fromarray(obs_render)
    img.save(f"./demo_frames/frame_0000.png")
    
    print("Reset → pose:", obs["pose"])
    for t in range(1, args.steps+1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t:02d} → reward={reward:.4f}, pose={obs['pose']}")
        print(
            f"history_poses: {obs['history_poses'].shape}, "
            f"excluding_poses: {obs['excluding_poses'].shape}"
        )
        if terminated or truncated:
            print("→ Episode done.")
            break
        obs_render = env.render()
        img = Image.fromarray(obs_render)
        img.save(f"./demo_frames/frame_{t:04d}.png")

    env.close()
