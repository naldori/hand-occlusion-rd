"""Stage B inference: public Hamba on DexYCB crops (GT 2D-joint bbox, demo's crop recipe), no detector.

Saves per image: predicted vertices/joints aligned two ways (PROTOCOL v3)
  B1  rigid (rotation + translation) alignment of wrist + 5 MCP joints to GT  (optimistic upper bound)
  B2  predicted orientation, translation re-solved from predicted 2D joints and the real intrinsics (no GT)
plus a PA-MPJPE sanity value. Run inside hamba_env with cwd = hand_pilot/hamba_run (for ./_DATA).
"""
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.environ.get("HAMBA_REPO", os.path.expanduser("~/Hamba")))

import cv2
import numpy as np
import torch
from skimage.filters import gaussian

from hamba.models import load_hamba
from hamba.datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2

ROOT = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot"))
CKPT = f"{ROOT}/ckpt/hamba/checkpoints/hamba.ckpt"
RES = f"{ROOT}/results/" + os.environ.get("PILOT_SET", "stage_b")
LABELS = f"{RES}/labels.json"
OUT = os.environ.get("PRED_OUT", f"{RES}/pred.npz")
RESCALE = 2.0                      # demo default
ALIGN_IDX = [0, 2, 5, 9, 13, 17]   # wrist + 5 MCP (OpenPose order; thumb MCP = 2)
BATCH = 16


def crop(img, box, cfg):
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
    scale = RESCALE * np.array([box[2] - box[0], box[3] - box[1]], np.float32) / 200.0
    bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=cfg.MODEL.BBOX_SHAPE).max()
    size = cfg.MODEL.IMAGE_SIZE
    cvimg = img.copy()
    down = bbox_size / size / 2.0
    if down > 1.1:
        cvimg = gaussian(cvimg, sigma=(down - 1) / 2, channel_axis=2, preserve_range=True)
    patch, _ = generate_image_patch_cv2(cvimg, center[0], center[1], bbox_size, bbox_size, size, size,
                                        False, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
    patch = convert_cvimg_to_tensor(patch[:, :, ::-1])
    mean, std = 255. * np.array(cfg.MODEL.IMAGE_MEAN), 255. * np.array(cfg.MODEL.IMAGE_STD)
    for c in range(3):
        patch[c] = (patch[c] - mean[c]) / std[c]
    return patch, center, bbox_size


def kabsch(src, dst, scale=False):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(s0.T @ d0)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (s0 ** 2).sum() if scale else 1.0
    return lambda x: s * (x - mu_s) @ R.T + mu_d


def solve_translation(J, uv, K):
    """t minimizing reprojection of (J + t) onto uv with intrinsics K (linear least squares)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A, b = [], []
    for (X, Y, Z), (u, v) in zip(J, uv):
        A.append([fx, 0, -(u - cx)]); b.append((u - cx) * Z - fx * X)
        A.append([0, fy, -(v - cy)]); b.append((v - cy) * Z - fy * Y)
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def main():
    model, cfg = load_hamba(CKPT)
    model = model.cuda().eval()
    rows = json.load(open(LABELS))
    images = {}
    for r in rows:
        images.setdefault(r['image'], r)
    keys = list(images)
    out = {k: [] for k in ['image', 'v_b1', 'v_b2', 'j_b1', 'j_b2', 'pa_mpjpe_mm', 'mpjpe_b2_mm',
                           'global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_cam_t', 'kp2d_full', 't_b2']}
    for s in range(0, len(keys), BATCH):
        chunk = keys[s:s + BATCH]
        patches, meta = [], []
        for k in chunk:
            r = images[k]
            p, c, b = crop(cv2.imread(k), r['box'], cfg)
            patches.append(torch.from_numpy(p).float()); meta.append((c, b, r))
        with torch.no_grad():
            o = model({'img': torch.stack(patches).cuda()})
        V = o['pred_vertices'].float().cpu().numpy()
        J = o['pred_keypoints_3d'].float().cpu().numpy()
        kp2d = o['pred_keypoints_2d'].float().cpu().numpy()
        mp = {k: v.float().cpu().numpy() for k, v in o['pred_mano_params'].items()}
        cam, cam_t = o['pred_cam'].float().cpu().numpy(), o['pred_cam_t'].float().cpu().numpy()
        for i, (c, b, r) in enumerate(meta):
            if 'gt_j21' in r:
                gt = np.array(r['gt_j21'], np.float64)
            else:
                gt = np.load(r['image'].replace('color_', 'labels_').replace('.jpg', '.npz'))['joint_3d'][0].astype(np.float64)
            f1 = kabsch(J[i][ALIGN_IDX], gt[ALIGN_IDX])
            uv = c + kp2d[i] * b
            t = solve_translation(J[i], uv, np.array(r['K']))
            out['image'].append(r['image'])
            out['v_b1'].append(f1(V[i])); out['j_b1'].append(f1(J[i]))
            out['v_b2'].append(V[i] + t); out['j_b2'].append(J[i] + t)
            out['pa_mpjpe_mm'].append(np.linalg.norm(kabsch(J[i], gt, scale=True)(J[i]) - gt, axis=1).mean() * 1000)
            out['mpjpe_b2_mm'].append(np.linalg.norm(J[i] + t - gt, axis=1).mean() * 1000)
            for k in ['global_orient', 'hand_pose', 'betas']:
                out[k].append(mp[k][i])
            out['pred_cam'].append(cam[i]); out['pred_cam_t'].append(cam_t[i])
            out['kp2d_full'].append(uv); out['t_b2'].append(t)
        print(f"{s + len(chunk)}/{len(keys)}  PA-MPJPE median so far {np.median(out['pa_mpjpe_mm']):.1f} mm", flush=True)
    np.savez_compressed(OUT, **{k: np.array(v) for k, v in out.items()})
    print("PA-MPJPE mm: median %.2f mean %.2f | B2 MPJPE mm: median %.1f" % (
        np.median(out['pa_mpjpe_mm']), np.mean(out['pa_mpjpe_mm']), np.median(out['mpjpe_b2_mm'])))


if __name__ == "__main__":
    main()
