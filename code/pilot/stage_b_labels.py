"""Stage B labels (PROTOCOL v3, items 5 and 6): per (image, finger) GT labels, no model involved.

- occlusion class per finger from GT meshes by ray casting camera -> joint:
  'object' if any of the finger's last 3 joints is hit by an object mesh in front of it,
  else 'self' if hit by another finger / the palm >= 15 mm in front of it, else 'visible'.
  Agreement with DexYCB seg labels is reported as a check.
- GT contact per finger at 2 / 3 / 5 mm (grasped object, watertight mesh).
- GT hand bbox from joint_2d, intrinsics, image path, for running the estimator.
"""
import glob
import json
import os

import numpy as np
import trimesh
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_a as A  # noqa: E402

OUT = f"{A.ROOT}/results/stage_b"
SELF_OCC_M = 0.015
FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_JOINTS = {f: [4 * i + 2, 4 * i + 3, 4 * i + 4] for i, f in enumerate(FINGERS)}   # OpenPose order
THRESH_MM = [2, 3, 5]


def intrinsics(serial):
    c = yaml.load(open(f"{A.DATA}/calibration/intrinsics/{serial}_640x480.yml"), Loader=yaml.FullLoader)['color']
    return np.array([[c['fx'], 0, c['ppx']], [0, c['fy'], c['ppy']], [0, 0, 1.0]])


def first_hits(mesh, targets):
    """distance along camera rays to every hit, per target point (camera at origin)."""
    dirs = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(np.zeros_like(dirs), dirs, multiple_hits=True)
    return np.linalg.norm(locs, axis=1), ray_idx, tri_idx


def main():
    os.makedirs(OUT, exist_ok=True)
    layer = A.mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    finger_of_v = np.array([A.FINGER_OF_PART[p] or 'palm' for p in part_of_v])
    faces = layer.th_faces.numpy()
    finger_of_face = finger_of_v[faces[:, 0]]
    frames = json.load(open(f"{A.ROOT}/results/stage_a_3mm/frames.json"))
    meshes, rows, agree = {}, [], []
    for fr in frames:
        seq = f"{A.DATA}/{A.SUBJECT}/{fr['seq']}"
        meta = yaml.load(open(f"{seq}/meta.yml"), Loader=yaml.FullLoader)
        betas = yaml.load(open(f"{A.DATA}/calibration/mano_{meta['mano_calib'][0]}/mano.yml"),
                          Loader=yaml.FullLoader)['betas']
        for serial in meta['serials']:
            lab = np.load(f"{seq}/{serial}/labels_{fr['frame']:06d}.npz")
            if np.all(lab['pose_m'] == 0):
                continue
            v, j21 = A.hand_mesh(layer, lab['pose_m'][0], betas)
            hand = trimesh.Trimesh(v, faces, process=False)
            objs = []
            for k, oid in enumerate(meta['ycb_ids']):
                if oid not in meshes:
                    meshes[oid] = A.load_object(oid)
                if np.any(lab['pose_y'][k]):
                    m = meshes[oid].copy()
                    m.apply_transform(np.vstack([lab['pose_y'][k], [0, 0, 0, 1]]))
                    objs.append((oid, m))
            grasped = objs[[o for o, _ in objs].index(meta['ycb_ids'][meta['ycb_grasp_ind']])][1]
            depth = np.linalg.norm(j21, axis=1)
            obj_occ = np.zeros(21, bool)
            for _, m in objs:
                d, ri, _ = first_hits(m, j21)
                obj_occ[np.unique(ri[d < depth[ri]])] = True
            d, ri, ti = first_hits(hand, j21)
            self_occ = np.zeros(21, bool)
            joint_finger = np.array(['palm'] + [f for f in FINGERS for _ in range(4)])
            m_self = (d < depth[ri] - SELF_OCC_M) & (finger_of_face[ti] != joint_finger[ri])
            self_occ[np.unique(ri[m_self])] = True
            # seg agreement check for object occlusion
            uv = np.round(lab['joint_2d'][0]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
            seg_obj = np.zeros(21, bool)
            seg_obj[inside] = np.isin(lab['seg'][uv[inside, 1], uv[inside, 0]], meta['ycb_ids'])
            agree += list(seg_obj[inside] == obj_occ[inside])
            # GT contact per finger at several thresholds
            closest, dist, _ = __import__('trimesh').proximity.closest_point(grasped, v)
            touch_in = grasped.contains(v)
            kp = lab['joint_2d'][0]
            box = [float(kp[:, 0].min()), float(kp[:, 1].min()), float(kp[:, 0].max()), float(kp[:, 1].max())]
            for f in FINGERS:
                jj = FINGER_JOINTS[f]
                occ = 'object' if obj_occ[jj].any() else ('self' if self_occ[jj].any() else 'visible')
                fv = finger_of_v == f
                rows.append(dict(seq=fr['seq'], frame=fr['frame'], serial=serial, finger=f, occlusion=occ,
                                 joints_in_image=bool(inside[jj].all()),
                                 **{f"gt_contact_{t}mm": bool((touch_in[fv] | (dist[fv] <= t / 1000)).any())
                                    for t in THRESH_MM},
                                 gt_tip=j21[jj[-1]].tolist(), box=box,
                                 image=f"{seq}/{serial}/color_{fr['frame']:06d}.jpg",
                                 K=intrinsics(serial).tolist()))
        print(fr['seq'], fr['frame'], len(rows), flush=True)
    json.dump(rows, open(f"{OUT}/labels.json", "w"))
    occ = [r['occlusion'] for r in rows]
    summary = dict(n_images=len(rows) // 5, n_fingers=len(rows),
                   occlusion_counts={k: occ.count(k) for k in ['visible', 'object', 'self']},
                   object_occ_vs_seg_agreement_per_joint=round(float(np.mean(agree)), 3),
                   gt_contact_rate={t: round(float(np.mean([r[f'gt_contact_{t}mm'] for r in rows])), 3)
                                    for t in THRESH_MM})
    json.dump(summary, open(f"{OUT}/labels_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
