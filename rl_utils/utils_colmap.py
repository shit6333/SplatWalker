import os
import struct
import numpy as np

def quaternion_to_rotation_matrix(qvec):
    """
    Convert quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = qvec
    # Normalize quaternion
    n = np.linalg.norm(qvec)
    if n > 0.0:
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def read_images_binary(path):
    """
    Read COLMAP images.bin and return a dict of camera-to-world poses.
    """
    Hs = []
    Ns = []
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('<I', f.read(4))[0]
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            cam_id = struct.unpack('<I', f.read(4))[0]
            # read name
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name_bytes.append(c)
            name = b''.join(name_bytes).decode('utf-8')
            # skip 2D points
            n_pts2d = struct.unpack('<Q', f.read(8))[0]
            f.read(n_pts2d * (8+8+8))
            # compute C2W
            R_wc = quaternion_to_rotation_matrix(qvec)
            R_cw = R_wc.T
            t_cw = -R_cw.dot(tvec)
            H = np.eye(4)
            H[:3,:3] = R_cw
            H[:3, 3] = t_cw
            Hs.append(H), Ns.append(name)
    return Hs, Ns



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/mnt/HDD3/miayan/omega/RL/gaussian-splatting/data/room/sparse/0',help='Directory containing cameras.bin and images.bin')
    args = parser.parse_args()
    
    # cam_file = os.path.join(model_dir, 'cameras.bin')
    img_file = os.path.join(args.model_dir, 'images.bin')
    Hs, Ns = read_images_binary(img_file)


    # print C2W pose
    # print(f"\nImage {first_id} ('{img_info['name']}') C2W Pose (4x4):")
    # print(img_info['c2w'])
    print(Ns[0])
    print(Hs[0])
    
    H = Hs[0]
    R = H[:3,:3]
    T = H[:3, 3]
    print(f'R : {R}, T : {T}')
    H_inv = np.linalg.inv(H)
    R_inv = H_inv[:3,:3]
    T_inv = H_inv[:3, 3]
    print(f'R : {R_inv}, T : {T_inv}')
