# 충전 리플맵 4.2 — 분석 파이프라인

2026 데이터+AI 혁신 챌린지(데이터안심구역 부문) 제출용. 스펙 원문: [docs/spec_리플맵4.2.md](docs/spec_리플맵4.2.md)

주민의 전력 이용 안정성·충전 접근성·합리적 공공 투자를 지원하는 **공공 충전 인프라 투자 우선순위 지도**를 목표로 한다. 현재 산출물은 부하 중심 잠정 검토표다. 상권 파급효과(DDD·이벤트 스터디·CATE)는 부가 편익으로 두고, 과거 이력 기반 부하 예측과 ESS 시뮬레이션으로 현장 검토 근거를 만든다. 저CATE 지역을 투자 후보에서 배제하지 않는다.

**현재 기준은 [v7 HTML](docs/충전리플맵4_2-07_08절-전력공학축확장안.html)과 [v7 구현 명세](docs/claude_claude-code-구현-프롬프트-리플맵4_2-08절추가모듈.md)다.** 원본 스펙과 다르면 이 개정안을 따른다. 실측 변압기 정격·주거지 접속관계·취약계층 수혜를 확보한 분석이 아니며, 정전 예방·교체비 절감은 입증한 성과가 아니다.

[최종 검증 기록](docs/최종검증_2026-09-17.md): 선택 패키지를 포함한 전체 테스트 49개 성공·실패 0개, 공학 시나리오 2,520행. 원자료 확보 후 확인할 항목도 기록했다.

> ⚠ `data/mock/` 은 전부 **가상** 데이터다. 로더·파이프라인 동작 검증용이며 결론에 쓰지 않는다.

## 실행

로컬(PowerShell, 저장소 루트):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe mock_data.py --out data/mock --energy
.venv\Scripts\python.exe kill_criteria.py --config config/mock.json
.venv\Scripts\python.exe pipeline.py --config config/mock.json
.venv\Scripts\python.exe -m pytest -q tests
```

현장(안심구역 JupyterLab):

1. `config/field_template.json` → `config/field.json`, `config/industry_codes_template.json` → `config/industry_codes.json` 복사.
2. `field.json` 의 `paths` 를 실제 경로로 바꾸고, 헤더가 다르면 `columns` 에 그 항목만 적는다(접두 일치 허용).
3. **첫날:** `from kill_criteria import run_checks; run_checks("config/field.json")` — 업종 코드가 비어 있어도 돈다.
4. `TB_SHC_TOBU_CODE.csv` · `TB_SHC_CODE.csv` 로 `industry_codes.json` 을 채운 뒤 `from pipeline import run; run("config/field.json")`.

5. 8-A/8-B는 날짜가 보존된 1시간 원자료와 최소 12주 연속 이력이 필요하다. `energy.evaluation_start`·공휴일 달력·후보 장치 사양을 자료에 맞춰 확인한다. 월 집계만 있으면 예측은 실패로 남기며 실측으로 대체하지 않는다.
6. SMP는 `timestamp,smp`(KST 시간 시작, 원/kWh) CSV로 정규화한 뒤 `paths.smp`를 지정한다. 미입력은 피크 목적함수와 비용 결측이다. 과거 SMP는 사후 가격 평가이며 사전 이용 가능 시점은 별도 확인한다.
7. 현재 신청에서 제외한 KEP_007은 현장 템플릿의 `null`을 유지한다. 해당 선택 단계 생략은 의도한 부분완료이며, mock 전체 실행은 기존 경로 회귀 검증을 위해 합성 KEP_007을 포함한다.

산출물: `out_dir/png/`(반출용) · `out_dir/csv/`(현장 작업용, 반출 대상 아님) · `out_dir/pipeline.log`.

## 반입 파일

| 묶음 | 내용 | 만드는 법 |
|---|---|---|
| `dist/ev-ripplemap_반입_<커밋>.zip` | 해당 커밋의 실행 코드(예측·ESS 포함) · 설정 · README · requirements · SHA-256 목록 | 새 커밋 기준 재생성 필요. 기존 ZIP은 v7 모듈을 포함하지 않음 |
| `dist/ev-ripplemap_참고자료_반입.zip` | 법정동코드 마스터 · 법정동 코드대응 · 법정동 중심점 · 대조표 · 안내 | `tools\build_reference_files.py` |

`config/field_template.json` 의 `paths` 가 참고자료 파일 이름을 그대로 가리킨다. 반입하지 않는 것: `mock_data.py` · `tests/` · `config/mock.json` · `config/industry_codes_mock.json` · `docs/` · `tools/`.

### 법정동 코드 개편 — 기준코드

마스터(2026-07)와 경계(2023-07) 사이에 법정동 코드가 크게 바뀌었다. 분석 데이터는 2023~2025 라 원천마다 개편 전/후 코드·명칭이 섞일 수 있다.

| 변경일 | 개편 | 읍면동 | 기준코드(2025-12-31 유효) |
|---|---|---:|---|
| 2023-12-29 | 부천시 → 원미·소사·오정구 | 24 | 신코드 |
| 2026-02-01 | 화성시 → 만세·효행·병점·동탄구 | 36 | 구코드 |
| 2026-06-30 | 인천 중·동·서구 → 제물포·영종·서해·검단구 | 80 | 구코드 |
| 2026-06-30 | 광주(29)·전남(46) → 전남광주통합특별시(12) | 623 | 구코드 |

- 마스터에는 신·구 코드를 모두 넣어 KEPCO 텍스트가 어느 명칭이든 매칭된다.
- `paths.bjd_crosswalk` 코드대응표로 KEPCO · SHC001 · SHC002 · 중심점 코드를 기준코드 하나로 묶는다(시계열 단절 방지). 연쇄 대응은 거부한다.
- 이름까지 바뀐 5곳(양지면→양지읍 등)은 자동 대응하지 못해 `중심점_마스터_대조.csv` 에 남는다.

## 구조

| 단계 | 파일 | 주요 산출물 | 실패 시 |
|---|---|---|---|
| 0 지역키 정합 | `bjd_mapping.py` | `s0_bjd_match_rate` · `s0_bjd_unmatched` | 중단 |
| 1 시계열 집계 | `kepco_loader.py` | `s1_kepco_monthly` · `s1_kepco_band_share` | 중단 |
| 2 변화점 | `kepco_loader.py` | `s2_activation` · `s2_activation_examples` | 중단 |
| 3 처치 정제 | `can_loader.py` · `kep007_loader.py` | `s3_can_*` · `s3_treated_excl_base` · `s3_kep007_stock` | **격리**(`s3_can_skipped`) |
| 4 Y_ddd | `identification.py` | `s4_ydd_panel` · `s4_skipped_*` · `s4_shc002_quality` | 중단 |
| 5 이벤트 스터디 | `identification.py` | `s5_event_main` · `s5_pretrend_*` · `s5_event_regression` · `s5_s6_event_study` | 중단 |
| 6 위약 | `identification.py` | `s6_event_placebo` · `s6_discount_by_k` · `s6_post_summary` | 중단 |
| 6.5 처치오염 | `diagnostics.py` | `s65_contamination` | 격리 |
| 7 CATE | `heterogeneity.py` | `s7_features` · `s7_cate_region` · `s7_cate_validation` · `s7_subgroup_cells`/`s7_importance` | 격리 |
| 8 처방 | `load_axis.py` | `s8_load_concentration` · `s8_quadrants` · `s8_quadrants_plot` | 격리 |
| 8-A 일별 예측 | `load_forecast.py` · `kepco_loader.py` | `s8a_validation` | 격리·실패 표시, 실측 대체 금지 |
| 8-B ESS | `ess_optimizer.py` | `s8b_scenarios` · `s8b_schedules` · `s8b_comparison` · `s9_public_review` | 격리·행별 제약/예측/가격 상태 기록 |
| 9 반출 | `outputs.py` · `pipeline.py` | `s0_run_summary` · `s9_manifest` | — |
| 킬 크라이테리아 | `kill_criteria.py` | `kill_criteria/png/k_kill_criteria` | 항목별 `오류` |

## 폴백 경로 (현장에 없을 수 있는 패키지)

| 패키지 | 있으면 | 없으면 |
|---|---|---|
| ruptures | `ruptures.Pelt(l2)` | `pelt_builtin` — 같은 목적함수의 정확해(월 36점이라 O(n²)) |
| econml | `CausalForestDML` | `estimate_cate_fallback` — 업종구성 × 외지유입 2×2 |
| scipy | 초과분 고정 방전·비첨두 충전 LP | 그리디 + 모든 물리 제약 재검증(최적해 아님) |
| linearmodels · statsmodels | 사용 안 함 | Callaway–Sant'Anna · FE 회귀 · 클러스터 SE · p값을 numpy 로 내장 |

## 설계 결정 — 스펙과 다르게 했거나 스펙이 정하지 않은 것 (팀 확인 필요)

| 항목 | 구현 | 이유 / 확인할 것 |
|---|---|---|
| **위약 Y 구조** | 대조 업종을 둘로 나눠 `placebo` 코드(예: 병원)를 대기소비 자리, 나머지 대조를 비교 자리에 넣은 같은 DDD | 스펙 문구대로 대기소비 자리에 대조 업종을 넣으면 Y = [대조] − [대조] ≡ 0 이라 검정이 성립하지 않는다. `placebo` 를 비우면 대조 업종 DD(외지−거주자)로 대체 |
| CS 추정기 | 패키지 없이 직접 구현(기준시점 g−1 고정, never-treated 대조, 법정동 클러스터 부트스트랩) | 주·위약을 같은 부트스트랩 가중치로 돌려 할인 후 SE 를 쌍으로 계산 |
| 폴백 회귀 | 처치군 + never-treated(있으면), 상대시점 더미 k=−1 제외·양 끝 binning, 2-way within | never-treated 가 없으면 공선성 때문에 최소 k 더미도 뺀다 |
| KEPCO 시각 | `조회기간`(YYYYMMDDHH) · 별도 시각 열 · 시간대별 가로 열 모두 지원(`layout=auto`) | 6개 컬럼 정의서만으로 시각 위치 불명. 1~24 표기는 자동으로 0~23 |
| 활성화 판정 | PELT(log 일평균) → ±2개월 안 전후 3개월 평균비 최대 달로 보정 → 2개월 안에 붙은 상승은 하나로 | [9/15 mock] 보정 없으면 18곳 중 5곳이 2개월 이르게 잡혔다. 보정 후 18/18 |
| never-treated | `control_lookback_start`(2024-07) 이후 상승 변화 없음 | 2023~24 자연 성장은 대조군 자격에 영향 없음 |
| 부하 집중도 | 1시간 단위(24개)로 계산, TIZO 구간 아님 | 구간 폭이 다르면 kWh 점유율이 폭에 비례해 부푼다. 값은 **구간 평균 kW** (순간 최대 아님) |
| SHC 코드 | `UMD_CD` 10자리 / 8자리 / `SGNG_CD`5+`UMD_CD`3 / 2+3+3 모두 수용 | VARCHAR2(10) 저장 형태 현장 확인 |
| 셀 결측 | 원천 행 없음 → 0(업종 부재), 행은 있는데 마스킹 → 결측 스킵. 셀 내 마스킹 행 50% 초과도 결측 | `max_masked_share` |
| 개설률 | `open_mode`: `flow`(신규 상태코드) / `stock_diff`(재고 순증 + 해지) | FRNC_STAT_CD '등록'이 흐름인지 재고인지 불명 |
| 처치오염 문턱 | 초과비율 ≥ 2.0 **그리고** 개설률 +2%p | [9/15 mock] 1.5 · 0.5%p 는 Poisson 잡음만으로 17곳 중 3곳 오탐 |
| 법정동별 사전추세 | 사전 변화 벡터의 호텔링 T² 예측검정, p값은 F(p, n−p) 정확분포 | [9/15 mock] 월별 z² 합 χ² → 5곳 오탐(같은 기준월을 빼 z 가 상관), 대조군 공분산 + χ² 근사 → 3곳(대조 10곳으로 4차원 공분산 추정 시 T² 부풂), F 정확분포 → 0곳 |
| CAN 판정 | 짧은 간격(≤10분) 연속 레코드의 순간이동(>200km/h)·SOC 급변(>15%p) 비율 | 이상치 뚜렷하면 식별번호 수와 무관하게 `model`. 문턱은 mock 기준 — 실데이터 근거표 보고 조정 |
| CAN → 법정동 | 법정동 **중심점 최근접** 근사(`emd_centroids`, QGIS) | 폴리곤 공간조인 결과가 있으면 교체 |
| CAN 피처 시점 | `can_feature_before`(2025-03) 이전 세션만 | 스펙은 "2022.11~ 오염 없음"이지만 데이터가 2025 이후까지 이어지면 bad control |
| CATE 단위결과 | Δ = 사후 3개월 평균 − 사전 평균, 대조군은 코호트 시점 분포 가중 | 대조군에도 셀 CATE 를 배정해 미활성 지역 처방에 쓴다 |
| CATE 검증 | 같은 층화 K-fold 에서 상수 ATE / 2×2 / CF 의 변환결과 MSE · GATES 비교 | CF 는 교차검증 손실이 2×2 보다 크면 채택하지 않음 |
| 4사분면 문턱 | CATE·집중도 모두 중앙값 | `load_axis.cate_cut`/`conc_cut` (`"zero"` 또는 숫자 가능) |
| 001/002 | 합산하지 않음. 활성화 시점은 `kepco.source`(제공 설정 002), 전력축은 001 | 실제 포괄 범위·사업자 채널 대응은 현장 명세 확인 |

## 공학 산출물 해석

- 상한은 평가 이전 28일 기존 충전 부하 최대값의 1.1/1.2/1.3배다. 변압기 용량 추정값이 아니다. 증설은 0/2/3/5대, 이용률은 CAN 상대 모양 또는 설정 모양에 가정 배율을 적용한다.
- ESS는 이산 후보에서 초기·종단 에너지, SOC 사용 범위, 충방전 효율·출력, 재충전, 순부하 0 이상을 검증해 선택한다. 전역 최소 비용 용량이 아니다. 하루씩 독립된 실험이며 열화·설비비를 제외한다.
- forecast와 oracle에 같은 사전 선정 용량을 사용한다. 예측 오차로 실측 적용 시 상한·종단잔량을 못 지키면 그대로 보고한다. SMP 비용 차이는 실제 전기요금·교체비 절감이 아니다.
- 공공성 증빙 미입력은 “자료 보완 후 검토”다. 낮은 필요도로 점수화하지 않는다. 부하 점검 순서는 미적용 초과 kWh → 잔여 초과 kWh 내림차순, 동률은 법정동코드다. 현장 증빙·비용효과 없이 투자 확정 순위로 쓰지 않는다.
- 코드 마스터는 도형이 아니다. 실제 2D 지도는 [국토교통부 경계 WFS](https://www.data.go.kr/data/15059008/openapi.do) 등의 기준일·좌표계·코드 정합을 확인한 후 QGIS에서 작성한다. 현재 Python 결과는 표·차트이며 지도 완성을 주장하지 않는다.
- 충전기 시간 이동 최적화는 도착·출차·충전 필요량 등 서비스 제약을 확보한 뒤 수행할 후속 기능이다.

## 반출 규칙 (코드로 강제)

- 모든 표는 CSV + PNG 동시 생성. PNG 가 반출용.
- 위도·경도·좌표 계열 컬럼이 든 표는 `OutputWriter` 가 저장을 거부(`ValueError`). CAN·KEP_007 좌표는 내부 계산에만 쓴다.
- CAN 법정동 표는 차량/세션 수 3 미만 칸을 `—` 로 억제.
- 경로 B 산출물 제목에는 "정황상 보조 근거, 개별 차량 식별 아님"을 붙인다.
- CATE 피처에 부하·집중도·피크 계열 이름이 들어오면 `ValueError`, 스냅샷이 활성화 창 시작 이후 달을 포함하면 `ValueError`.

## mock 검증 결과 (9/15, seed 42)

| 확인 | 심은 값 | 결과 |
|---|---|---|
| 법정동 매칭 | 31곳 중 미매칭 1(행복동), '제1동' 표기 6, 동명이인 '중앙동' | 원문 24 · 정규화 6 · 미매칭 1 (3.2%) |
| 활성화 시점 | 처치 18 · never 10 · 창 이전 1 · 창 이후 1 | 18/18 월 단위 일치, 상태 분류 전부 일치 |
| 주 효과(사후 평균) | 평균 τ 0.117 | CS 0.106 (SE 0.036) · 회귀 0.110 (SE 0.032) · 사전추세 p=0.26 |
| 위약 | 0 | 0.023 (SE 0.021) → 할인 후 0.083 |
| 처치오염 | 2곳 | 정확히 2곳 유보 |
| 법정동별 사전추세 | 0곳 | 유보 0/18 (합성 패널에 심은 추세 1곳은 검출 — 테스트) |
| CAN | individual / model 파일 | individual → 경로 A, model → 경로 B |
| 킬 크라이테리아 | 후보 18곳 · 002⊆001 | 1 경고(선택 패키지) · 2 경고(30곳 미만) · 3~8 통과, 13초 |
| 통합 테스트 | — | 25 passed · pyflakes 0 |

4사분면 유보 12곳 중 10곳은 'CAN 거점성 충전 다수'다 — mock 에서 자가충전 거점을 12개 법정동에 몰아 둔 탓이며, 실데이터에서 이 비율이 높으면 `can.base_share_flag` 기준을 재검토할 것.

9/15에는 Causal Forest가 미검증이었으나, 9/17에 별도 Python 3.10·EconML 0.17.0 환경에서 120개 합성 지역(처치 60개)의 실제 학습·교차검증·효과 추정 경로를 검증했다. 기본 30지역 mock은 처치 표본 문턱 미달로 여전히 2×2 폴백이 정상이다. 실제 자료에서 Causal Forest의 성능이 검증됐다는 뜻은 아니다. 현장에서는 `s7_cate_validation`을 확인한다.
