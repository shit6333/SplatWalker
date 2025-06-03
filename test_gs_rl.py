import os
from pathlib import Path
import torch
from torchvision.utils import save_image
from env import UnifiedGaussianEnv
from aesthetics_model import AestheticsModel
from drqv2_net import DrQV2Agent
import cv2
from glob import glob
from camera_interpolation import interpolate_poses

@torch.no_grad()
def test_agent(
    ply_path: str,
    ckpt_path: str,
    out_dir: str = "./demo_test_frames",
    max_steps: int = 200,
    init_idx: int = -1,
    device: str = "cuda",
    images_bin: str = './image.bin',
    z_rotate=False,
    interpolation = False,
    interpolation_path = None,
    img_width=1557, img_height=1038, fx=1586.0, fy=1586.0
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # build environment
    aes_model = AestheticsModel(device=device)
    env = UnifiedGaussianEnv(
        ply_path=ply_path,
        aes_model=aes_model,
        img_width=img_width,
        img_height=img_height,
        fx=fx,
        fy=fy,
        smooth_window=3,
        history_length=3,
        excluding_length=3,
        device=device,
        sh_degree=3,
        images_bin = images_bin, # colmap images.bin path
        z_rotate = z_rotate
    )

    obs_shape = env.observation_space["image"].shape
    if z_rotate:
        pose_shape = [6]
    else:
        pose_shape = [5]
    action_shape = env.action_space.shape

    # init agent & load weights
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        pos_shape=pose_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=10000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,1000000)',
        stddev_clip=0.3,
        use_tb=False,
        use_context=False,
        context_hidden_dim=128,
        context_history_length=5,
        nstep=1,
        batch_size=256,
        num_scenes=1,
        use_position=True,
        diversity=True,
        exc_hidden_size=128,
        no_hidden=False,
        num_excluding_sequences=env.excluding_length,
        order_invariant=False,
        distance_obs=False,
        smoothness=True,
        position_only_smoothness=False,
        smoothness_window=3,
        position_orientation_separate=False,
        rand_diversity_radius=False,
        constant_noise=-1,
        no_aug=False,
    )

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.set_weights(ckpt['agent_state_dict'])

    # testing
    if init_idx != -1:
        obs_dict, _ = env.reset(idx=init_idx)
    else:
        obs_dict, _ = env.reset()
    obs_img = obs_dict['image']
    
    if z_rotate:
            obs_pose = obs_dict['pose']
            history_poses = obs_dict['history_poses']
            excluding_poses = obs_dict['excluding_poses']
    else:
        obs_pose = obs_dict['pose'][:-1]
        history_poses = obs_dict['history_poses'][:,:-1]
        excluding_poses = obs_dict['excluding_poses'][:,:-1]

    frame_idx = 0
    vis_img = env._render()
    save_image(torch.from_numpy(vis_img), f"{out_dir}/frame_{frame_idx:04d}.png")
    pose_record = []
    if interpolation:
        pose_record.append([env.R.copy(), env.T.copy()])

    done = False
    t_step = 0
    avg_aes = 0
    avg_smoothness = 0
    
    
    while not done and t_step < max_steps:
        action = agent.act(
            obs=obs_img,
            pos=obs_pose,
            t=t_step,
            excluding_seq=excluding_poses,
            avg_step_size=history_poses,
            step=-1, 
            eval_mode=True
        )
        # action = env.action_space.sample() # random sample

        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs_img = obs_dict['image']
        if z_rotate:
            obs_pose = obs_dict['pose']
            history_poses = obs_dict['history_poses']
            excluding_poses = obs_dict['excluding_poses']
        else:
            obs_pose = obs_dict['pose'][:-1]
            history_poses = obs_dict['history_poses'][:,:-1]
            excluding_poses = obs_dict['excluding_poses'][:,:-1]

        frame_idx += 1
        vis_img = env._render()
        save_image(torch.from_numpy(vis_img), f"{out_dir}/frame_{frame_idx:04d}.png")
        if interpolation:
            pose_record.append([env.R.copy(), env.T.copy()])

        done = terminated or truncated
        t_step += 1
        aes = info['aesthetic']
        smo = info['smoothness']
        avg_aes += aes
        avg_smoothness += smo
        print(f'Step: {t_step} save to => f"{out_dir}/frame_{frame_idx:04d}.png" | Reward: {reward} | Aes: {aes}')

    print(f"Test finished. Total steps: {t_step}")
    
    # interpolate camer pose
    if interpolation:
        print(f"Interpolate The Camera Pose...")
        inter_step = 1
        for i in range(len(pose_record)-1):
            R_t1, T_t1 = pose_record[i]
            R_t2, T_t2 = pose_record[i+1]
            Rs_mid, Ts_mid = interpolate_poses(R_t1, T_t1, R_t2, T_t2, n=15)
            
            vis_img = env._render_by_pose(R_t1, T_t1)
            save_image(torch.from_numpy(vis_img), f"{interpolation_path}/frame_{inter_step:04d}.png")
            inter_step += 1
            
            for i in range(len(Rs_mid)):
                vis_img = env._render_by_pose(Rs_mid[i], Ts_mid[i])
                save_image(torch.from_numpy(vis_img), f"{interpolation_path}/frame_{inter_step:04d}.png")
                inter_step += 1
    env.close()
    
    avg_aes = avg_aes / t_step
    avg_smoothness = avg_smoothness / t_step
    return avg_aes, avg_smoothness

def images_to_video(image_folder, output_path, fps=30):
    # Get all image paths and sort them by filename
    image_paths = sorted(glob(os.path.join(image_folder, '*.png')))

    if not image_paths:
        raise ValueError("No PNG images found in the specified folder.")

    # Read the first image to get the frame size
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: could not read image {img_path}. Skipping.")
            continue
        out.write(frame)  # Write frame to video

    out.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    # set init camera pose (select from training frame)
    init_frame_idxs = [1,50,100,150,200,250,300,310,30,135,175,235,275,290]
    
    # set the testing pretrain weights
    model_names = ['checkpoints_myroom_step15_aes3_rn_var_penalty',
                   'checkpoints_garden_step15_aes3_rn_var_penalty_new']
    
    # params
    z_rotate = False    # include z axis rotation in action space
    interpolation = True # make camera interpolation
    epoch = 48000 
    
    for init_frame_idx in init_frame_idxs:
        for model_name in model_names:
            parser = argparse.ArgumentParser()
            parser.add_argument("--ply", type=str, default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/output/room_midnerf/point_cloud/iteration_30000/point_cloud.ply", help="Path to .ply file")
            parser.add_argument("--images_bin", type=str, default="/mnt/HDD3/miayan/omega/RL/gaussian-splatting/data/room/sparse/0/images.bin", help="Bin file for camera params")
            parser.add_argument("--ckpt", type=str, default=f"/mnt/HDD3/miayan/omega/RL/gaussian-splatting/{model_name}/agent_ep{epoch}.pth", help="Path to model checkpoint (.pth)")
            parser.add_argument("--outdir", type=str, default="./demo_test_frames")
            parser.add_argument("--interpolate_outdir", type=str, default="./demo_test_frames_inter")
            parser.add_argument("--max_steps", type=int, default=15)
            parser.add_argument("--device", type=str, default="cuda")
            parser.add_argument("--video_path", type=str, default=f'./videos/{model_name}/room_video{init_frame_idx}.mp4')
            parser.add_argument("--inter_video_path", type=str, default=f'./videos/{model_name}/room_video{init_frame_idx}_inter.mp4')
            parser.add_argument("--img_width", type=float, default=1557.0, help="Max steps per episode")
            parser.add_argument("--img_height", type=float, default=1038, help="Max steps per episode")
            parser.add_argument("--fx", type=float, default=1586.0, help="Max steps per episode")
            parser.add_argument("--fy", type=float, default=1586.0, help="Max steps per episode")
            args = parser.parse_args()
            os.makedirs(f'./videos/{model_name}', exist_ok=True)
            
            # make camera intepolation for smooth video
            if interpolation:
                avg_aes, avg_smoothness = test_agent(
                    ply_path=args.ply,
                    ckpt_path=args.ckpt,
                    out_dir=args.outdir,
                    max_steps=args.max_steps,
                    device=args.device,
                    init_idx = init_frame_idx,
                    images_bin = args.images_bin,
                    z_rotate = z_rotate,
                    interpolation = True,
                    interpolation_path = args.interpolate_outdir,
                    img_width=args.img_width, 
                    img_height=args.img_height, 
                    fx=args.fx, 
                    fy=args.fy
                )
                print(f'AVG AES:{avg_aes} / AVG SMO: {avg_smoothness}')
                images_to_video(args.interpolate_outdir, args.inter_video_path, fps=30)
            else:
                # for testing sparse results
                avg_aes, avg_smoothness = test_agent(
                    ply_path=args.ply,
                    ckpt_path=args.ckpt,
                    out_dir=args.outdir,
                    max_steps=args.max_steps,
                    device=args.device,
                    init_idx = init_frame_idx,
                    images_bin = args.images_bin,
                    z_rotate = z_rotate,
                    img_width=args.img_width, 
                    img_height=args.img_height, 
                    fx=args.fx, 
                    fy=args.fy
                )
                print(f'AVG AES:{avg_aes} / AVG SMO: {avg_smoothness}')
                images_to_video(args.outdir, args.video_path, fps=2)