"""HO-Cap version of stage_b_labels (rules in PROTOCOL.md, 'HO-Cap 적용 규칙')."""
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_a as A  # noqa: E402
from stage_b_infer_utils import hits_before, kabsch, to_local  # noqa: E402
from stage_b_labels import FINGERS, FINGER_JOINTS, SELF_OCC_M, THRESH_MM, first_hits  # noqa: E402

D = f"{A.ROOT}/data_hocap"
OUT = f"{A.ROOT}/results/stage_b_hocap"
MAX_FIT_RES_M = 0.001
FIT_IDX = [i for i in range(21) if i != 8]      # all joints except the index tip


def main():
    os.makedirs(OUT, exist_ok=True)
    layer = A.mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    finger_of_v = np.array([A.FINGER_OF_PART[p] or 'palm' for p in part_of_v])
    faces = layer.th_faces.numpy()
    finger_of_face = finger_of_v[faces[:, 0]]
    joint_finger = np.array(['palm'] + [f for f in FINGERS for _ in range(4)])
    betas = yaml.safe_load(open(f"{D}/calibration/mano/subject_5.yaml"))['betas']
    rows, agree, fit_res, dropped = [], [], [], 0
    for Q in sorted(glob.glob(f"{D}/subject_5/2*/")):
        seq = os.path.basename(Q.rstrip('/'))
        meta = yaml.safe_load(open(Q + 'meta.yaml'))
        serials = meta['realsense']['serials']
        oids = meta['object_ids']
        meshes = {o: trimesh.load(f"{D}/models/{o}/cleaned_mesh_10000.obj", force='mesh') for o in oids}
        pm = np.load(Q + 'poses_m.npy')[0]                     # right hand
        ref = {}                                                # object positions at frame 0 (camera 0)
        for fi in range(0, meta['num_frames'], 5):
            lab0 = np.load(f"{Q}{serials[0]}/label_{fi:06d}.npz")
            if (lab0['hand_joints_2d'][0] < 0).any() or np.allclose(pm[fi], -1):
                continue
            if not ref:
                first = np.load(f"{Q}{serials[0]}/label_000000.npz")['obj_poses']
                ref = {o: first[k][:3, 3] for k, o in enumerate(oids)}
            v_w, j_w = A.hand_mesh(layer, pm[fi], betas)          # world frame
            # grasped object from camera 0
            f0 = kabsch(j_w[FIT_IDX], lab0['hand_joints_3d'][0][FIT_IDX].astype(np.float64))
            v0 = f0(v_w)
            best, best_n = None, 0
            for k, o in enumerate(oids):
                m, vl = meshes[o], to_local(v0, lab0['obj_poses'][k])
                _, d, _ = trimesh.proximity.closest_point(m, vl)
                n = int(((d <= 0.003) | m.contains(vl)).sum())
                if n > best_n:
                    best, best_n = k, n
            if best is None or np.linalg.norm(lab0['obj_poses'][best][:3, 3] - ref[oids[best]]) < A.LIFT_M:
                continue
            for serial in serials:
                lab = np.load(f"{Q}{serial}/label_{fi:06d}.npz")
                if (lab['hand_joints_2d'][0] < 0).any():
                    continue
                gt = lab['hand_joints_3d'][0].astype(np.float64)
                f = kabsch(j_w[FIT_IDX], gt[FIT_IDX])
                res = float(np.linalg.norm(f(j_w[FIT_IDX]) - gt[FIT_IDX], axis=1).max())
                fit_res.append(res)
                if res > MAX_FIT_RES_M:
                    dropped += 1
                    continue
                v = f(v_w)
                gt[8] = f(j_w)[8]          # index tip: MANO standard vertex (HO-Cap label uses another vertex)
                hand = trimesh.Trimesh(v, faces, process=False)
                depth = np.linalg.norm(gt, axis=1)
                obj_occ = np.zeros(21, bool)
                for k, o in enumerate(oids):
                    d, ri, _ = hits_before(meshes[o], lab['obj_poses'][k], gt)
                    obj_occ[np.unique(ri[d < depth[ri]])] = True
                d, ri, ti = first_hits(hand, gt)
                self_occ = np.zeros(21, bool)
                self_occ[np.unique(ri[(d < depth[ri] - SELF_OCC_M) & (finger_of_face[ti] != joint_finger[ri])])] = True
                uv = lab['hand_joints_2d'][0]
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
                seg_obj = np.zeros(21, bool)
                seg_obj[inside] = np.isin(lab['seg_mask'][uv[inside, 1], uv[inside, 0]], [1, 2, 3, 4])
                agree += list(seg_obj[inside] == obj_occ[inside])
                grasped, vl = meshes[oids[best]], to_local(v, lab['obj_poses'][best])
                _, dist, _ = trimesh.proximity.closest_point(grasped, vl)
                touch_in = grasped.contains(vl)
                box = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
                for fg in FINGERS:
                    jj = FINGER_JOINTS[fg]
                    occ = 'object' if obj_occ[jj].any() else ('self' if self_occ[jj].any() else 'visible')
                    fv = finger_of_v == fg
                    rows.append(dict(seq=f"{seq}|{oids[best]}", sequence=seq, frame=fi, serial=serial, finger=fg,
                                     occlusion=occ, joints_in_image=bool(inside[jj].all()),
                                     **{f"gt_contact_{t}mm": bool((touch_in[fv] | (dist[fv] <= t / 1000)).any())
                                        for t in THRESH_MM},
                                     gt_tip=gt[jj[-1]].tolist(), gt_j21=gt.tolist(), box=box,
                                     image=f"{Q}{serial}/color_{fi:06d}.jpg", K=lab['cam_K'].tolist(),
                                     obj_mesh=f"{D}/models/{oids[best]}/cleaned_mesh_10000.obj",
                                     obj_pose=lab['obj_poses'][best].tolist()))
            print(seq, fi, oids[best], len(rows), flush=True)
    json.dump(rows, open(f"{OUT}/labels.json", "w"))
    occ = [r['occlusion'] for r in rows]
    summary = dict(n_images=len(rows) // 5, n_fingers=len(rows), n_episodes=len({r['seq'] for r in rows}),
                   episodes=sorted({r['seq'] for r in rows}),
                   occlusion_counts={k: occ.count(k) for k in ['visible', 'object', 'self']},
                   object_occ_vs_seg_agreement_per_joint=round(float(np.mean(agree)), 3),
                   gt_fit_residual_mm_max=round(max(fit_res) * 1000, 3), images_dropped_fit=dropped,
                   gt_contact_rate={t: round(float(np.mean([r[f'gt_contact_{t}mm'] for r in rows])), 3)
                                    for t in THRESH_MM})
    json.dump(summary, open(f"{OUT}/labels_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
