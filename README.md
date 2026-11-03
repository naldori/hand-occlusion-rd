# hand-occlusion-pilot

Hamba를 베이스로 **범용 3D 손 자세 추정기**를 개발하기 위한 R&D 작업 저장소다. 목표 추정기는 물체를 쥔 손에서도 무너지지 않아야 하고,
이후 hand-object / human-object pose estimation과 sim2real2sim 파이프라인(사람 시연 → 시뮬레이션 → 로봇 → 실세계)의 입력으로 쓰인다.
이번 라운드(2026-11-02 ~ 11-03)는 HaMeR 계열 추정기(Hamba, WiLoR)의 **물체 가림 실패 모드**를 실제 데이터로 진단하고,
물리·기하 모듈을 덧붙여 오류를 찾거나 고칠 수 있는지 go/no-go를 판단했다.

- **결론과 다음 작업** → [`REPORT.md`](REPORT.md) (요약, 단계별 결과, 로드맵 시사점, 우선순위 백로그)
- **사전 고정한 go/no-go 기준과 변경 이력** → [`PROTOCOL.md`](PROTOCOL.md) (실행 순서대로의 원본 기록, 표현만 정리)

핵심 결과: 물리(평형)·기하(관통) 검사는 가린 손가락 오류 탐지기로 no-go. 배포 조건 오차의 대부분은 손 전체의 미터 깊이이며,
두 모델 모두 평균 손 크기를 가정한다. 물체를 미터 기준으로 쓰는 평행이동 보정은 손목 깊이 오차를 50–57% 줄였고(go),
정답 가림 라벨 대신 공개 학습형 접촉 예측기(HACO)를 써도 같은 효과가 났다(C-learned). 가린 손가락 단위 오류 검출은 아직 기준에 근소하게 못 미친다.

## 폴더 구조
```
REPORT.md                전체 수행 내역과 결론
PROTOCOL.md              사전 고정 기준, 변경 이력, 단계별 결과(시간순 원본 기록)
code/
  toy_physics/           가상 실험 (이미지 없음, 원기둥 + 점 접촉 평형 LP)
    check_equilibrium.py   E1 계산 시간, E2 접촉 수별 평형 가능 비율, E3 손가락 오류 검출
    score_sensitivity.py   연속 점수(필요한 최대 힘)의 오류 민감도
    cause_check.py         접촉 수 × 힘 상한별 검출률 (주의: import하면 본문 실험이 다시 실행됨)
    min_force_check.py     "닿은 손가락은 최소 힘으로 민다" 가정
  pilot/                 실제 데이터 파일럿 (DexYCB, HO-Cap) — 단계 순서
    stage_a.py             단계 A: GT만으로 파지 유형 통계 + GT 힘 검사 타당성 (CONTACT_MM=2|3|5)
    stage_b_labels.py      단계 B 라벨(DexYCB): 손가락별 가림 3분류(ray casting), GT 접촉
    hocap_labels.py        단계 B 라벨(HO-Cap)
    stage_b_infer.py       Hamba 추론(GT bbox crop) + B1/B2 정렬 저장   [hamba_env, cwd=hamba_run]
    load_check.py          Hamba 체크포인트가 키 누락 없이 로드되는지 검증
    stage_b_infer_utils.py Kabsch 정렬, 평행이동 최소제곱, 물체 좌표계 변환
    stage_b_score.py       S_geo / S_geo-oracle, 층별 AUROC + 시퀀스 bootstrap   (PILOT_SET=...)
    diagnostics.py         Diagnostics D1 오차 방향, D2 굽힘·평균 자세, D3 물체 기준 위치
    diagnostics_figures.py
    diagnostics2_ab.py     Diagnostics 2 A("떠 있음" 원인), B(깊이 오차 원인)
    diagnostics2_c.py      Diagnostics 2 C(물체를 자로 쓰는 평행이동 보정, C_MODE=grid|apply)
    diagnostics2_figures.py
    wilor_infer.py         WiLoR 추론(같은 입력·정렬)   [wilor_env, cwd=WiLoR repo]
    wilor_compare.py       Hamba vs WiLoR 비교 표·그림
    haco_infer.py          HACO 정점별 접촉 추론(같은 GT bbox, 공식 데모 전처리)   [haco_env, cwd=HACO repo]
    c_learned_stage1.py    C-learned 단계 1: 손가락 접촉 정확도(HACO vs 대리 규칙)
    c_learned_stage2.py    C-learned 단계 2: 깊이 보정 비교 표(C-none/proxy/learned/oracle, C*)
    c_learned_detect.py    C-learned 단계 3: S_geo-learned 오류 검출(B1/B2/보정 후, HAND_MODEL=hamba|wilor)
    c_learned_figures.py
    run_*.sh               실행 순서를 묶은 셸 스크립트(당시 사용 그대로)
results/                 요약 결과만(표·JSON·그림). 행 단위 원자료는 아래 "원자료 위치" 참고
  toy_physics/  stage_a_{2,3,5}mm/  stage_b/  stage_b_hocap/  diagnostics/  diagnostics2/  wilor_compare/  c_learned/
env/
  requirements_env.txt        CPU 분석용 환경(pip freeze)
  requirements_hamba_env.txt  Hamba 추론 환경(torch 2.1.2+cu118, VMamba selective_scan sm_89 빌드)
  requirements_wilor_env.txt  WiLoR 추론 환경(hamba_env 복제)
  requirements_haco_env.txt   HACO 추론 환경(공식 권장 Python 3.8, torch 1.11+cu113)
  external_repos.txt          사용한 외부 저장소와 커밋
```

## 재현 방법 (요약)
1. **경로**: 모든 스크립트는 `HAND_PILOT_ROOT`(기본 `~/hand_pilot`) 아래에 다음이 있다고 가정한다.
   `mano/MANO_RIGHT.pkl`, `mano_models/`(같은 파일 링크), `data/`(DexYCB subject-01, calibration, models),
   `data_hocap/`(HO-Cap subject_5, labels, models, poses, calibration), `ckpt/hamba/`, `hamba_run/_DATA/`, `WiLoR/`,
   `HACO/`(공식 저장소 + `release_checkpoint/haco_neurips_hamer_checkpoint.ckpt` + 공식 demo `data/base_data`),
   그리고 이 저장소의 `code/pilot/*`가 `$HAND_PILOT_ROOT/code/`에 있어야 한다(셸 스크립트가 그 위치를 씀).
   Hamba 저장소 위치는 `HAMBA_REPO`(기본 `~/Hamba`).
2. **데이터**: MANO는 mano.is.tue.mpg.de에서 직접(등록 필요). DexYCB는 dex-ycb.github.io(subject-01 12GB + models 1.4GB + calibration).
   HO-Cap은 irvlutd.github.io/HOCap(subject_5 3.2GB + labels 1.5GB + models/poses/calibration, 등록 불필요).
3. **환경**: `env/requirements_*.txt`. Hamba는 detectron2·ViTPose·Mamba 없이 VMamba `kernels/selective_scan`만 빌드
   (`setuptools<70`, `numpy==1.23.5`(chumpy 호환), CUDA 11.8, `TORCH_CUDA_ARCH_LIST=8.9`). 모든 실행은 `PYTHONNOUSERSITE=1`.
4. **순서**: stage_a → stage_b_labels / hocap_labels → stage_b_infer → stage_b_score → diagnostics(+figures)
   → diagnostics2_ab → diagnostics2_c(grid, apply) → diagnostics2_figures → wilor_infer → stage_b_score / diagnostics / diagnostics2_c(proxy) → wilor_compare
   → haco_infer → c_learned_stage1 → diagnostics2_c(proxy,learned) → c_learned_stage2 → c_learned_detect → c_learned_figures.
   구체적인 환경 변수 조합은 `code/pilot/run_*.sh`와 REPORT.md의 "재현" 절 참고.

## 원자료 위치 (저장소에 넣지 않음, 수 MB~수십 MB)
`$HAND_PILOT_ROOT/results/` 아래: `*/labels.json`(손가락별 라벨), `*/finger_scores.json`(손가락별 점수),
`diagnostics*/**/rows.json · a_rows.json · c_rows.json`(진단 원자료), `diagnostics/repro_*/pred.npz`, `wilor_compare/*/pred.npz`(모델 예측),
`c_learned/*/haco_contact.npz · finger_contact.json`(HACO 정점·손가락 접촉), `c_learned/**/c_rows.json`(보정 원자료).

## 저장소 사본에 대한 주의
`code/`는 실행에 쓴 스크립트의 사본이며, 실행 환경의 고정 경로를 환경 변수(`HAND_PILOT_ROOT`, `HAMBA_REPO`, 기본값 `~/hand_pilot`, `~/Hamba`)로 바꾼 것 외에는 동일하다.
이 사본은 문법 검사와 import 시험만 했고, 전체 파이프라인을 사본으로 다시 돌리지는 않았다.

## 라이선스
이 저장소의 코드·문서·그림·결과 요약은 [CC BY-NC 4.0](LICENSE)(출처 표시, **비상업적 이용만 허용**)을 따른다.
참조하는 외부 자료(DexYCB, HO-Cap, MANO, Hamba·WiLoR·VMamba·HACO 코드와 가중치)는 이 저장소에 포함되지 않으며 각자의 라이선스를 따른다
(HACO는 CC BY-NC 4.0).
