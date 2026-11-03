"""Load public Hamba checkpoint and verify every weight was actually loaded (load_hamba uses strict=False)."""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.environ.get("HAMBA_REPO", os.path.expanduser("~/Hamba")))
import torch
from hamba.models import load_hamba
CKPT = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot")) + "/ckpt/hamba/checkpoints/hamba.ckpt"
model, cfg = load_hamba(CKPT)
sd = torch.load(CKPT, map_location="cpu")["state_dict"]
msd = model.state_dict()
missing = [k for k in msd if k not in sd]
unexpected = [k for k in sd if k not in msd]
shape_bad = [k for k in msd if k in sd and sd[k].shape != msd[k].shape]
print("missing", len(missing), missing[:10]); print("unexpected", len(unexpected), unexpected[:10]); print("shape mismatch", shape_bad[:10])
print("init_hand_pose nonzero (from ckpt, not placeholder):", bool(model.mano_head.init_hand_pose.abs().sum() > 0))
print("params(M):", sum(p.numel() for p in model.parameters()) / 1e6)
