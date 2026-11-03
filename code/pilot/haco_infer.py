"""HACO per-vertex contact on the same GT-bbox crops (PROTOCOL.md 'C-learned').
Official demo preprocessing; only the detector box is replaced by the GT 2D-joint box.
Run with haco_env, cwd = HACO repo:  PILOT_SET=stage_b_hocap python haco_infer.py
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.transforms import Normalize  # noqa: E402

from lib.core.config import cfg  # noqa: E402
from lib.models.model import HACO  # noqa: E402
from lib.utils.preprocessing import augmentation_contact  # noqa: E402

ROOT = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot"))
SET = os.environ.get("PILOT_SET", "stage_b_hocap")
OUT = f"{ROOT}/results/c_learned/{SET}"
CKPT = "release_checkpoint/haco_neurips_hamer_checkpoint.ckpt"
BATCH = 16


def main():
    os.makedirs(OUT, exist_ok=True)
    model = HACO().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location='cuda')['state_dict'])
    norm = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
    rows = json.load(open(f"{ROOT}/results/{SET}/labels.json"))
    boxes = {}
    for r in rows:
        boxes.setdefault(r['image'], r['box'])
    keys = list(boxes)
    probs = []
    for s in range(0, len(keys), BATCH):
        batch = []
        for k in keys[s:s + BATCH]:
            img = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)
            x0, y0, x1, y1 = boxes[k]
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            r_ = cfg.DATASET.ho_big_bbox_expand_ratio            # 2.0, as in the official image demo
            bbox = [cx - 0.5 * r_ * w, cy - 0.5 * r_ * h, r_ * w, r_ * h]
            crop, *_ = augmentation_contact(img.copy(), bbox, 'test', enforce_flip=False)
            batch.append(norm(torch.from_numpy(crop.transpose(2, 0, 1) / 255.0)).float())
        with torch.no_grad():
            o = model({'input': {'image': torch.stack(batch).cuda()}}, mode='test')
        probs.append(o['contact_out'].sigmoid().cpu().numpy())
        print(f"{s + len(batch)}/{len(keys)}", flush=True)
    np.savez_compressed(f"{OUT}/haco_contact.npz", image=np.array(keys), prob=np.concatenate(probs).astype(np.float32))
    print("saved", OUT, np.concatenate(probs).shape, "ckpt_expand_ratio", cfg.DATASET.ho_big_bbox_expand_ratio)


if __name__ == "__main__":
    main()
