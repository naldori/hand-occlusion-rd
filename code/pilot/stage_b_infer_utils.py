import numpy as np


def kabsch(src, dst, scale=False):
    """returns f(x) applying the rigid (optionally similarity) transform that maps src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(s0.T @ d0)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (s0 ** 2).sum() if scale else 1.0
    return lambda x: s * (x - mu_s) @ R.T + mu_d


def to_local(points, pose):
    """camera-frame points -> object frame of a rigid pose (4x4 or 3x4). Distances/inside tests are invariant."""
    pose = np.asarray(pose)
    return (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]


def hits_before(mesh_local, pose, targets):
    """For camera rays (camera at origin) toward each target: distances of hits on a posed mesh,
    computed in the mesh's own frame so the mesh (and its acceleration structures) is never copied."""
    o = to_local(np.zeros((1, 3)), pose)[0]
    tl = to_local(targets, pose)
    dirs = tl - o
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    locs, ray_idx, tri_idx = mesh_local.ray.intersects_location(np.repeat(o[None], len(tl), 0), dirs, multiple_hits=True)
    return np.linalg.norm(locs - o, axis=1), ray_idx, tri_idx


def solve_translation(J, uv, K):
    """t minimizing reprojection of (J + t) onto uv with intrinsics K (linear least squares)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A, b = [], []
    for (X, Y, Z), (u, v) in zip(J, uv):
        A.append([fx, 0, -(u - cx)]); b.append((u - cx) * Z - fx * X)
        A.append([0, fy, -(v - cy)]); b.append((v - cy) * Z - fy * Y)
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
