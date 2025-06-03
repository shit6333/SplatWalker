# AestheticCam

## 1. Environment

```
Python Version : 3.10.16
```

The Environment can follow the [Official 3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting) to install, and use the `pip install -r requirements.txt` to install remaining modules

## 2. Data prepare

### 2-1. Aesthetic Model Weight
1. [Download the aesthetic model weights](https://drive.google.com/drive/u/0/folders/14uI-h2KAQQrT1rc8KNjRQNay-ghaQXlF) and place them in the following directory:
`pretrain/model_params/`
2. In `aesthetic_model.py`, update the `self.resume` variable to the path of your downloaded model file:

```python
self.resume = ['pretrain/model_params/your_model.pth.tar']
```

### 2-2. 3DGS Scene Setup
We provide a [pretrained 3DGS weight](https://drive.google.com/drive/u/0/folders/1OTEJ6r04woJUPcsFyWTHlSc6RDdVOBei) trained on the **room** scene from the **Mip-NeRF** dataset. You can use this directly for training, or follow the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) codebase to reconstruct your own scenes.

**How to Use Your Own 3DGS Scene :**
1. Place your pretrained `.ply` file in the appropriate directory, and modify the argument in `train_gs_rl.py`:
```python
parser.add_argument("--ply", type=str, default="/output/room_midnerf/point_cloud/iteration_30000/point_cloud.ply")
```
2. Specify the corresponding COLMAP images.bin file that contains camera pose information:
```python=
images_bin = '/data/room/sparse/0/images.bin'
```
- If this is not provided, the training will sample the initial camera pose from a normal distribution.

## 3. Code

- train_gs_rl.py : training
- test_gs_rl.py : testing 
- drqv2_net.py : RL Agent
- aesthetics_model.py : aesthetic model
- rl_utils/ : utils for rl

For testing demo we prepare the [model weight](https://drive.google.com/drive/u/0/folders/18dNuHwRC_u76fYyCDc8a0TqzVquNLBJI) train on the scene room for 40000 epochs. (replace the path in test_gs_rl.py)

## reference code:
> - [3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting)
> - [GAIT: Generating Aesthetic Indoor Tours with Deep Reinforcement Learning](https://github.com/desaixie/gait)
> - [Good View Hunting: Learning Photo Composition from Dense View Pairs](https://github.com/zijunwei/ViewEvaluationNet)