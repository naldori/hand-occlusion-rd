"""Stage A (PROTOCOL v2): grasp-type statistics and force-check sanity on DexYCB GT only.

Per lifted frame: GT MANO hand mesh + GT YCB object mesh (camera frame of one camera),
contact parts (16 MANO skinning parts, <=3mm or penetrating), precision-grasp flag (<=3 parts),
and static-equilibrium feasibility of the GT grasp (mu=0.5, 0.5N <= normal force <= 15N).
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
import yaml
from manopth.manolayer import ManoLayer
from scipy.optimize import linprog

ROOT = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot"))
DATA = f"{ROOT}/data"
SUBJECT = "20200709-subject-01"
OUT = f"{ROOT}/results/stage_a_" + os.environ.get("CONTACT_MM", "3") + "mm"
SERIAL_IND = 0            # one camera per sequence (contacts are view independent)
FRAME_STEP = 5
LIFT_M = 0.05
CONTACT_M = float(os.environ.get("CONTACT_MM", "3")) / 1000
MU, FMIN, FMAX, N_EDGES = 0.5, 0.5, 15.0, 8
L_CHAR = 0.05             # torque scaling [m]
SEED = 0

YCB = {1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box', 4: '005_tomato_soup_can',
       5: '006_mustard_bottle', 6: '007_tuna_fish_can', 7: '008_pudding_box', 8: '009_gelatin_box',
       9: '010_potted_meat_can', 10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
       13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block', 17: '037_scissors',
       18: '040_large_marker', 19: '051_large_clamp', 20: '052_extra_large_clamp', 21: '061_foam_brick'}
# approximate masses [kg] from the YCB object set paper (Calli et al. 2015)
MASS = {1: .414, 2: .411, 3: .514, 4: .349, 5: .603, 6: .171, 7: .187, 8: .097, 9: .370, 10: .066,
        11: .178, 12: 1.131, 13: .147, 14: .118, 15: .895, 16: .729, 17: .082, 18: .016, 19: .125,
        20: .102, 21: .028}
PART_NAMES = ['palm', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2',
              'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3']
FINGER_OF_PART = [None] + ['index'] * 3 + ['middle'] * 3 + ['pinky'] * 3 + ['ring'] * 3 + ['thumb'] * 3
TIP_VERTS = [745, 317, 444, 556, 673]      # thumb, index, middle, ring, pinky
MANO_TO_OP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def mano_layer():
    # manopth, as in dex-ycb-toolkit (smplx's PCA path did not reproduce DexYCB joint_3d)
    return ManoLayer(flat_hand_mean=False, ncomps=45, side='right',
                     mano_root=f"{ROOT}/mano_models", use_pca=True)


def hand_mesh(layer, pose_m, betas):
    p = torch.tensor(pose_m, dtype=torch.float32).reshape(1, 51)
    v, j = layer(p[:, :48], torch.tensor(betas, dtype=torch.float32).reshape(1, 10), p[:, 48:51])
    return v[0].numpy() / 1000, j[0].numpy() / 1000     # joints already in OpenPose order


def table_frame(meta):
    ext = yaml.load(open(f"{DATA}/calibration/extrinsics_{meta['extrinsics']}/extrinsics.yml"),
                    Loader=yaml.FullLoader)
    serial = meta['serials'][SERIAL_IND] if 'serials' in meta else sorted(ext['extrinsics'])[SERIAL_IND]
    T_cam = np.array(ext['extrinsics'][serial], dtype=np.float64).reshape(3, 4)   # cam -> master
    T_tag = np.array(ext['extrinsics']['apriltag'], dtype=np.float64).reshape(3, 4)            # tag -> master
    # table normal (tag z) expressed in camera frame
    n_cam = T_cam[:, :3].T @ T_tag[:, 2]
    o_cam = T_cam[:, :3].T @ (T_tag[:, 3] - T_cam[:, 3])
    return serial, n_cam / np.linalg.norm(n_cam), o_cam


def load_object(oid):
    """YCB textured_simple.obj is split at texture seams; merge vertices so it is watertight."""
    m = trimesh.load(f"{DATA}/models/{YCB[oid]}/textured_simple.obj", force='mesh', process=False)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    assert m.is_watertight, YCB[oid]
    return m


def contacts(hand_v, part_of_v, mesh):
    closest, dist, tri = trimesh.proximity.closest_point(mesh, hand_v)
    outward = mesh.face_normals[tri]
    inside = mesh.contains(hand_v)      # ray test; mesh is made watertight in load_object
    touch = inside | (dist <= CONTACT_M)
    pen = np.where(inside, dist, 0.0)
    parts = []
    for k in range(16):
        m = touch & (part_of_v == k)
        if m.any():
            n_in = -outward[m].mean(0)
            parts.append(dict(part=k, p=closest[m].mean(0), n=n_in / np.linalg.norm(n_in)))
    return parts, float(pen.max())


def cone(n):
    t1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, [1.0, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    a = np.linspace(0, 2 * np.pi, N_EDGES, endpoint=False)
    return [n + MU * (np.cos(x) * t1 + np.sin(x) * t2) for x in a]


def equilibrium_residual(parts, com, g_vec, fmin=FMIN):
    """min ||wrench residual||_1, forces in friction cones with fmin <= normal <= FMAX. 0 => feasible."""
    k = len(parts)
    if k == 0:
        return float('inf')
    cols = [np.r_[d, np.cross(c['p'] - com, d) / L_CHAR] for c in parts for d in cone(c['n'])]
    A = np.array(cols).T
    nv = A.shape[1]
    S = np.zeros((k, nv + 12))
    for i in range(k):
        S[i, i * N_EDGES:(i + 1) * N_EDGES] = 1
    r = linprog(np.r_[np.zeros(nv), np.ones(12)],
                A_ub=np.vstack([S, -S]), b_ub=np.r_[np.full(k, FMAX), np.full(k, -fmin)],
                A_eq=np.hstack([A, np.eye(6), -np.eye(6)]), b_eq=-np.r_[g_vec, np.zeros(3)],
                bounds=(0, None), method="highs")
    return float(r.fun) if r.status == 0 else float('inf')


def main():
    os.makedirs(OUT, exist_ok=True)
    layer = mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    meshes = {}
    seqs = sorted(glob.glob(f"{DATA}/{SUBJECT}/*/meta.yml"))
    seqs = [os.path.dirname(s) for i, s in enumerate(seqs) if i % 5 == 4]     # S0 val (held out from train)
    rows, max_joint_err = [], 0.0
    for seq in seqs:
        meta = yaml.load(open(f"{seq}/meta.yml"), Loader=yaml.FullLoader)
        if meta['mano_sides'][0] != 'right':
            continue
        betas = yaml.load(open(f"{DATA}/calibration/mano_{meta['mano_calib'][0]}/mano.yml"),
                          Loader=yaml.FullLoader)['betas']
        serial, n_tab, o_tab = table_frame(meta)
        gi = meta['ycb_grasp_ind']
        oid = meta['ycb_ids'][gi]
        if oid not in meshes:
            meshes[oid] = load_object(oid)
        mesh0 = meshes[oid]
        com_local = mesh0.center_mass if mesh0.is_watertight else mesh0.bounds.mean(0)
        labels = sorted(glob.glob(f"{seq}/{serial}/labels_*.npz"))
        h0 = None
        # sign of table normal: hand/object should be on the positive side
        for fi, lf in enumerate(labels):
            lab = np.load(lf)
            pose_y = lab['pose_y'][gi]
            obj_c = pose_y[:, :3] @ com_local + pose_y[:, 3]
            if fi == 0:
                if np.dot(obj_c - o_tab, n_tab) < 0:
                    n_tab = -n_tab
                h0 = np.dot(obj_c - o_tab, n_tab)
            if fi % FRAME_STEP:
                continue
            lift = np.dot(obj_c - o_tab, n_tab) - h0
            if lift < LIFT_M or np.all(lab['pose_m'] == 0):
                continue
            v, j21 = hand_mesh(layer, lab['pose_m'][0], betas)
            max_joint_err = max(max_joint_err, float(np.abs(j21 - lab['joint_3d'][0]).max()))
            obj_mesh = mesh0.copy()
            obj_mesh.apply_transform(np.vstack([pose_y, [0, 0, 0, 1]]))
            parts, pen = contacts(v, part_of_v, obj_mesh)
            g_vec = -MASS[oid] * 9.81 * n_tab
            res = equilibrium_residual(parts, obj_c, g_vec)
            res0 = equilibrium_residual(parts, obj_c, g_vec, fmin=0.0)
            fingers = sorted({FINGER_OF_PART[c['part']] for c in parts if c['part'] > 0})
            rows.append(dict(seq=os.path.basename(seq), frame=fi, object=YCB[oid], lift_m=round(float(lift), 4),
                             n_parts=len(parts), parts=[PART_NAMES[c['part']] for c in parts],
                             n_fingers=len(fingers), fingers=fingers, palm='palm' in [PART_NAMES[c['part']] for c in parts],
                             max_pen_mm=round(pen * 1000, 2), eq_res=res, eq_feasible=res < 1e-6,
                             eq_feasible_fmin0=res0 < 1e-6))
        print(os.path.basename(seq), YCB[oid], sum(r['seq'] == os.path.basename(seq) for r in rows), flush=True)
    json.dump(rows, open(f"{OUT}/frames.json", "w"), indent=1)
    summarize(rows, max_joint_err)


def cluster_boot(rows, stat, n=2000):
    rng = np.random.default_rng(SEED)
    seqs = sorted({r['seq'] for r in rows})
    by = {s: [r for r in rows if r['seq'] == s] for s in seqs}
    vals = []
    for _ in range(n):
        pick = rng.choice(seqs, len(seqs))
        rr = [r for s in pick for r in by[s]]
        vals.append(stat(rr))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def summarize(rows, max_joint_err):
    prec = lambda rr: np.mean([r['n_parts'] <= 3 for r in rr])
    feas = lambda rr: np.mean([r['eq_feasible'] for r in rr])
    s = dict(
        n_frames=len(rows), n_sequences=len({r['seq'] for r in rows}),
        smplx_vs_dexycb_joint_max_abs_err_mm=round(max_joint_err * 1000, 3),
        n_parts_hist={int(k): int(v) for k, v in zip(*np.unique([r['n_parts'] for r in rows], return_counts=True))},
        n_fingers_hist={int(k): int(v) for k, v in zip(*np.unique([r['n_fingers'] for r in rows], return_counts=True))},
        precision_rate=round(float(prec(rows)), 3), precision_rate_ci95_seqboot=cluster_boot(rows, prec),
        gt_force_check_pass_rate=round(float(feas(rows)), 3), gt_force_check_pass_ci95_seqboot=cluster_boot(rows, feas),
        gt_force_check_pass_rate_fmin0=round(float(np.mean([r['eq_feasible_fmin0'] for r in rows])), 3),
        median_max_penetration_mm=float(np.median([r['max_pen_mm'] for r in rows])),
        decision_precision_ge_20pct=bool(prec(rows) >= 0.2),
        decision_force_tool_valid=bool(feas(rows) >= 0.8),
    )
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
