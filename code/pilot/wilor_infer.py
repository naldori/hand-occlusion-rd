"""WiLoR inference with the same inputs/alignments as stage_b_infer.py (Hamba), for the model comparison.

Crop: WiLoR's own ViTDetDataset with the GT 2D-joint bbox (rescale 2.0 as in its demo; no WiLoR detector).
Camera -> original image:
  WiLoR predicts a weak-perspective crop camera (s, tx, ty) for a 256 px crop with focal 5000 px.
  Its 2D keypoints are in crop-normalized units [-0.5, 0.5]; full-image pixels = box_center + kp * bbox_size
  (bbox_size = crop side in original pixels). The demo converts to a full-image translation with
  focal 5000/256 * max(W, H) (cam_crop_to_full). We do not use that focal: B2 solves the translation by least squares
  from these full-image 2D keypoints with the real intrinsics K (identical to the Hamba B2 procedure);
  B1 rigidly aligns wrist + 5 MCP joints to GT.
Usage (cwd = WiLoR repo): PILOT_SET=<stage_b|stage_b_hocap> python wilor_infer.py
"""
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wilor.models import load_wilor  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from stage_b_infer_utils import kabsch, solve_translation  # noqa: E402

ROOT = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot"))
SET = os.environ.get("PILOT_SET", "stage_b_hocap")
OUT = f"{ROOT}/results/wilor_compare/{SET}"
ALIGN_IDX = [0, 2, 5, 9, 13, 17]
BATCH = 16


def main():
    os.makedirs(OUT, exist_ok=True)
    model, cfg = load_wilor('pretrained_models/wilor_final.ckpt', 'pretrained_models/model_config.yaml')
    model = model.cuda().eval()
    rows = json.load(open(f"{ROOT}/results/{SET}/labels.json"))
    if not os.path.exists(f"{OUT}/labels.json"):
        os.symlink(f"{ROOT}/results/{SET}/labels.json", f"{OUT}/labels.json")
    images = {}
    for r in rows:
        images.setdefault(r['image'], r)
    keys = list(images)
    out = {k: [] for k in ['image', 'v_b1', 'v_b2', 'j_b1', 'j_b2', 'pa_mpjpe_mm', 'mpjpe_b2_mm',
                           'global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_cam_t', 'kp2d_full', 't_b2']}
    for s in range(0, len(keys), BATCH):
        chunk = keys[s:s + BATCH]
        items = []
        for k in chunk:
            r = images[k]
            ds = ViTDetDataset(cfg, cv2.imread(k), np.array([r['box']], np.float32), np.array([1.0]), rescale_factor=2.0)
            items.append(ds[0])
        batch = {k: torch.as_tensor(np.stack([it[k] for it in items])).cuda() for k in ['img']}
        with torch.no_grad():
            o = model({'img': batch['img'].float()})
        V = o['pred_vertices'].float().cpu().numpy()
        J = o['pred_keypoints_3d'].float().cpu().numpy()
        kp2d = o['pred_keypoints_2d'].float().cpu().numpy()
        mp = {k: v.float().cpu().numpy() for k, v in o['pred_mano_params'].items()}
        cam, cam_t = o['pred_cam'].float().cpu().numpy(), o['pred_cam_t'].float().cpu().numpy()
        for i, k in enumerate(chunk):
            r, it = images[k], items[i]
            gt = np.array(r['gt_j21'], np.float64) if 'gt_j21' in r else \
                np.load(k.replace('color_', 'labels_').replace('.jpg', '.npz'))['joint_3d'][0].astype(np.float64)
            f1 = kabsch(J[i][ALIGN_IDX], gt[ALIGN_IDX])
            uv = np.array(it['box_center']) + kp2d[i] * float(it['box_size'])
            t = solve_translation(J[i], uv, np.array(r['K']))
            out['image'].append(k)
            out['v_b1'].append(f1(V[i])); out['j_b1'].append(f1(J[i]))
            out['v_b2'].append(V[i] + t); out['j_b2'].append(J[i] + t)
            out['pa_mpjpe_mm'].append(np.linalg.norm(kabsch(J[i], gt, scale=True)(J[i]) - gt, axis=1).mean() * 1000)
            out['mpjpe_b2_mm'].append(np.linalg.norm(J[i] + t - gt, axis=1).mean() * 1000)
            for kk in ['global_orient', 'hand_pose', 'betas']:
                out[kk].append(mp[kk][i])
            out['pred_cam'].append(cam[i]); out['pred_cam_t'].append(cam_t[i])
            out['kp2d_full'].append(uv); out['t_b2'].append(t)
    np.savez_compressed(f"{OUT}/pred.npz", **{k: np.array(v) for k, v in out.items()})
    print("%s WiLoR PA-MPJPE mm: median %.2f | B2 MPJPE mm: median %.1f" % (
        SET, np.median(out['pa_mpjpe_mm']), np.median(out['mpjpe_b2_mm'])))


if __name__ == "__main__":
    main()
