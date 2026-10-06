# Team Ant Observation Ablation

공통 Team1 environment에서 terrain representation을 비교하는 팀 실험 저장소입니다.
Stage 1은 stock 7-term reward, Stage 2는 explicit Contact observation과
Team1 v3_depth modified reward 조건에서 HeightScan/DepthCam을 비교합니다.

- [공통 실험 protocol 및 비교 표](experiments/observation_ablation/README.md)
- [고정 protocol JSON](experiments/observation_ablation/protocol.json)
- [Canonical HeightScan manifest: 4096×32×1000](experiments/observation_ablation/heightscan_4096x32x1000/manifest.json)
- [Previous exploratory HeightScan manifest: 2048×32×10000](experiments/observation_ablation/heightscan/manifest.json)
- [Stage 2 공통 protocol](experiments/observation_ablation/shared/stage2_contact_modified_protocol.json)
- [Stage 2 HeightScan arm manifest](experiments/observation_ablation/heightscan_contact_modified_4096x32x1000/manifest.json)
- [기존 Team1 프로젝트 설명](project.md)

`Ant-rl-Ablation-HeightScan-v0`는 기존 59-D proprio + canonical 63-D HeightScan을 사용합니다.
Base와 DepthCam stock-reward 실험은 추후 같은 protocol로 수행해야 합니다.
기존 `Ant-rl-v0`, Team1 custom reward, DepthCam checkpoint와 평가 artifact는 보존합니다.
기존 custom-reward 결과를 stock-reward ablation의 Base/DepthCam 결과로 간주하지 않습니다.
