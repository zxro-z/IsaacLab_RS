# Contact feedback comparison

HeightScan과 Stock Reward를 고정하고 explicit 4-D Contact observation만 추가한 비교다. 동일 seed의 단일 실험이므로 일반적인 통계적 유의성을 주장하지 않는다.

| Metric | HeightScan + Stock | HeightScan + Contact + Stock |
|---|---:|---:|
| Return | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 |
| Displacement (m) | 60.5140 ± 29.1880 | 61.8529 ± 27.8017 |
| Duration (s) | 12.1947 ± 5.5930 | 12.9088 ± 5.3955 |
| Mean vx (m/s) | 4.4506 ± 1.5331 | 4.3692 ± 1.3281 |
| Fall | 41/100 | 29/100 |
| Timeout | 59/100 | 71/100 |
| >=5m | 87/100 | 89/100 |
| Out-of-terrain-X | 28/100 | 27/100 |

| Component | HeightScan + Stock | HeightScan + Contact + Stock |
|---|---:|---:|
| progress | 60.4711 ± 29.1633 | 61.8339 ± 27.8095 |
| alive | 6.0939 ± 2.7998 | 6.4520 ± 2.7012 |
| upright | 1.0924 ± 0.5508 | 1.1876 ± 0.5511 |
| move_to_target | 5.3633 ± 2.6815 | 5.8745 ± 2.6763 |
| action_l2 | -1.2979 ± 8.7617 | -0.5343 ± 1.3955 |
| energy | -7.2101 ± 3.5063 | -7.6240 ± 3.3849 |
| joint_pos_limits | -3.1773 ± 1.6496 | -3.6215 ± 1.6988 |
| total | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 |

표준편차는 population std이다. 두 조건은 동일한 Stock Reward를 사용한다. Map-boundary termination이 없으므로 displacement와 progress에는 terrain X bounds 밖의 이동이 포함될 수 있다. 공통 README의 Stage 구조는 변경하지 않았다.
