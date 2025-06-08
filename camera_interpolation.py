import numpy as np

def _rotmat_to_quaternion(R: np.ndarray) -> np.ndarray:
    # calculate trace
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)  # 正規化

def _quaternion_to_rotmat(q: np.ndarray) -> np.ndarray:
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - z*w)
    R[0, 2] = 2*(x*z + y*w)
    R[1, 0] = 2*(x*y + z*w)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - x*w)
    R[2, 0] = 2*(x*z - y*w)
    R[2, 1] = 2*(y*z + x*w)
    R[2, 2] = 1 - 2*(x*x + y*y)
    return R

def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q1 + t*(q2 - q1)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return (s1 * q1) + (s2 * q2)

def interpolate_poses(
    R1: np.ndarray, T1: np.ndarray,
    R2: np.ndarray, T2: np.ndarray,
    n: int
) -> (np.ndarray, np.ndarray):
    q1 = _rotmat_to_quaternion(R1)
    q2 = _rotmat_to_quaternion(R2)
    ts = np.linspace(1.0/(n+1), n/(n+1), n, dtype=np.float64)

    Rs_interp = np.zeros((n, 3, 3), dtype=np.float64)
    Ts_interp = np.zeros((n, 3), dtype=np.float64)

    for idx, t in enumerate(ts):
        qi = _slerp(q1, q2, t)
        Ri = _quaternion_to_rotmat(qi)
        Rs_interp[idx] = Ri
        Ti = (1 - t) * T1 + t * T2
        Ts_interp[idx] = Ti

    return Rs_interp, Ts_interp
