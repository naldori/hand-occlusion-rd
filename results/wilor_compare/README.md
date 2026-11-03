# WiLoR 비교 (2026-11-03) — Hamba의 결론이 HaMeR 계열 공통인지
새 기준 없음. 기존 프로토콜(프레임, 가림 라벨, B1/B2 정렬, 큰 오류 15mm, S_geo 정의, 시퀀스 bootstrap)과 판정 기준을 그대로 사용.
Hamba 결과 파일은 읽기만 함.

## 준비
- WiLoR 공식 저장소(rolpotamias/WiLoR) + 공개 체크포인트 wilor_final.ckpt(HF). Hamba 환경 복제(wilor_env)에서 추가 설치 없이 동작.
  로드 검증: 키 누락·불일치·shape 불일치 0 (파라미터 6.41억).
- 학습 데이터(논문 Sec.10, 저장소 model_config.yaml): FreiHAND, HO3D, MTC, RHD, InterHand2.6M, H2O3D, **DexYCB**, COCO-WholeBody,
  Halpe, MPII NZSL, BEDLAM, ARCTIC, Re:InterHand, Hot3D. **HO-Cap 없음.** (WHIM은 검출기 학습에만 사용) → Hamba와 같은 본/안 본 조건.
- 입력: WiLoR 검출기 미사용. WiLoR의 ViTDetDataset에 Hamba 때와 같은 정답 2D 관절 bbox(확대 2.0, 192:256 비율, 256px crop).
- 카메라 → 원본 이미지 좌표: WiLoR는 256px crop에 대한 약원근 카메라(s, tx, ty)와 초점거리 5000px를 가정한다.
  2D 관절(crop 정규화 [−0.5, 0.5])을 원본 픽셀로: box_center + kp × bbox_size(crop 한 변의 원본 픽셀 길이).
  데모는 5000/256 × max(W, H) 환산 초점거리로 cam_crop_to_full을 쓰지만 여기서는 쓰지 않음.
  B2 = 위 원본 2D 관절과 실제 내부 파라미터 K로 평행이동 최소제곱 재풀이(Hamba와 동일), B1 = 손목+MCP 5개 강체 정렬.
- C-proxy 가중치는 Hamba-DexYCB에서 정한 w_c=10, w_p=1 그대로.

## 비교 표
| item | Hamba · HO-Cap subject_5 | WiLoR · HO-Cap subject_5 | Hamba · DexYCB subject-01 | WiLoR · DexYCB subject-01 |
|---|---|---|---|---|
| PA-MPJPE median (mm) | 5.1 | 5.1 | 3.0 | 3.3 |
| tip error median B1 (mm) | 12.8 | 12.2 | 6.5 | 7.2 |
| tip error median B2 (mm) | 57.3 | 48.2 | 18.2 | 15.4 |
| pred hand size median (mm) | 93.8 | 94.3 | 92.0 | 91.5 |
| pred hand size p5-p95 (mm) | 90.9 – 94.9 | 93.3 – 94.7 | 90.9 – 93.0 | 90.8 – 92.7 |
| GT hand size (mm) | 88.1 | 88.1 | 91.8 | 91.8 |
| pred vs GT size (%) | 6.5 | 7.1 | 0.26 | -0.30 |
| B2 wrist depth error median (mm) | 53.0 | 44.0 | 14.8 | 11.0 |
| B2 wrist-translation line-of-sight share (object-occl. large) | 0.99 | 0.98 | 0.99 | 0.97 |
| B2 wrist translation explains tip error (object-occl. large) | 0.87 | 0.84 | 0.81 | 0.68 |
| AUROC S_geo B1 object-occluded | 0.463 [0.41, 0.53] | 0.460 [0.39, 0.53] | 0.359 [0.27, 0.64] | 0.413 [0.29, 0.72] |
| AUROC S_geo-oracle B1 object-occluded | 0.599 [0.56, 0.64] | 0.572 [0.53, 0.62] | 0.569 [0.34, 0.69] | 0.570 [0.40, 0.74] |
| n large errors B1 object-occluded | 1722 | 1520 | 139 | 159 |
| AUROC S_geo B2 object-occluded | 0.276 [0.20, 0.37] | 0.275 [0.24, 0.32] | 0.398 [0.34, 0.46] | 0.551 [0.47, 0.66] |
| AUROC S_geo-oracle B2 object-occluded | 0.715 [0.65, 0.78] | 0.704 [0.65, 0.76] | 0.654 [0.54, 0.76] | 0.704 [0.61, 0.79] |
| n large errors B2 object-occluded | 3997 | 3869 | 640 | 544 |
| C-proxy wrist depth (mm) before -> after | 53.0 → 23.0 | 44.0 → 22.0 | 14.8 → 12.5 | 11.0 → 12.9 |
| C-proxy wrist depth reduction | 0.566 [0.43, 0.66] | 0.499 [0.36, 0.58] | 0.159 [-0.16, 0.41] | -0.175 [-0.57, 0.02] |
| C-proxy wrist depth change CI (mm) | -39.7 – -20.3 | -28.6 – -13.9 | -7.1 – 1.9 | -0.2 – 6.3 |
| H1 met (S_geo CI lower > 0.6, B1 or B2) | False | False | False | False |
| object tells depth met (C-proxy >=30% and CI excludes 0) | True | True |  |  |

## 판정 (기존 기준)
- H1 (실사용 S_geo AUROC 하한 > 0.6): WiLoR 미충족 (HO-Cap B1 0.46 [0.39, 0.53], B2 0.28 [0.24, 0.32]; DexYCB B1 0.41, B2 0.55 [0.47, 0.66]).
- 물체가 깊이를 알려 준다 (HO-Cap, C-proxy ≥30% 감소, CI∌0): WiLoR 충족 (44.0 → 22.0mm, −50% [36, 58], 변화 CI −28.6 ~ −13.9mm).

| 문장 | WiLoR에서도 성립? | 근거 |
|---|---|---|
| (a) 사람과 상관없이 평균에 가까운 손 크기를 가정한다 | 예 | subject_5(정답 88.1mm)에 94.3mm(+7.1%), 5–95% 범위 93.3–94.7로 Hamba보다 더 좁게 MANO 평균(94.7) 근처. 학습에서 본 subject-01은 −0.3%로 정확 |
| (b) 실사용 기하 점수는 가려진 손가락의 큰 오류를 골라내지 못한다 | 예 | 물체 가림 S_geo AUROC 0.28–0.55, 모든 칸에서 하한 < 0.6. 정답 접촉을 알 때만 0.57–0.70 |
| (c) 가림 라벨과 물체 정보가 있으면 물체를 자로 손 깊이를 크게 줄일 수 있다 | 예 (깊이 오차가 클 때) | HO-Cap −50% [36, 58]. 단 깊이 오차가 이미 작은 DexYCB(11mm)에서는 −18% [−57, 2]로 오히려 약간 나빠짐(Hamba DexYCB도 −16%, 유의하지 않음) |

## 결론
WiLoR는 Hamba와 기본 정확도(PA-MPJPE 5.1 / 3.3mm)가 거의 같고, 세 결론이 모두 그대로 재현됐다. 처음 보는 작은 손(subject_5)에서
두 모델 모두 손 크기를 평균 근처로 6.5–7.1% 크게 내고 그만큼 손 전체를 멀리 놓으며(B2 손목 깊이 오차 53 / 44mm), 실사용 기하 점수는
가려진 손가락의 큰 오류를 골라내지 못하고, 가림 라벨과 정답 물체로 평행이동을 보정하면 HO-Cap 손목 깊이 오차가 절반 정도로 준다.
따라서 이 결론들은 Hamba 고유가 아니라 HaMeR 계열(ViT 백본 + MANO 회귀, 고정 초점거리) 공통 특성으로 보는 것이 맞다.
한계: 모델 2개, 피험자 데이터셋당 1명, 물체 정답 사용(상한), 두 모델 모두 DexYCB를 학습에 사용, 사후 비교로 기존 판정에 영향 없음.

## 파일
compare.json, compare.md, fig_hand_size.png, stage_b_hocap/ · stage_b/ (pred.npz, finger_scores.json, auroc_summary.json,
diag/ (D1–D3), c/ (C-proxy)), c_weights_from_hamba.json. 코드: code/wilor_infer.py, code/wilor_compare.py
