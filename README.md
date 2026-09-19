# 충전 리플맵 4.2 — 분석 파이프라인

2026 데이터+AI 혁신 챌린지(데이터안심구역 부문) 제출용. 현재 설계 기준은 루트의 V9 HTML과 이 README다.

주민의 전력 이용 안정성·충전 접근성·합리적 공공 투자를 지원하는 **공공 충전 인프라 투자 검토 우선순위 지도**를 목표로 한다. V9은 정책 상한 초과량, 역방향 2SFCA 접근성 부족도, SMP 기반 ESS 운영비 절감 잠재력을 결합한다. 상권 파급효과(DDD·이벤트 스터디·CATE)는 부가 편익 참고값이며 결합점수에 넣지 않는다.

**현재 기준은 루트의 V9 HTML과 이 README다.** 실측 변압기 정격·주거지 접속관계·취약계층 수혜를 확보한 분석이 아니며, 정전 예방·교체비 절감은 입증한 성과가 아니다. 결합점수는 실제 설치 지점 선정 결과가 아니라 법정동별 현장 검토 순서다.

V9 현재 회귀 검증은 55개 성공·실패 0개·선택 패키지 2개 생략이다.

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

5. 8-A/8-B는 날짜가 보존된 1시간 원자료와 최소 12주 연속 이력이 필요하다. `energy.evaluation_start`는 현장에서 반드시 지정하고, 공휴일 달력·후보 장치 사양을 자료에 맞춰 확인한다. 로더는 검증·교정·평가에 필요한 기간만 읽는다. 월 집계만 있으면 예측은 실패로 남기며 실측으로 대체하지 않는다.
6. SMP는 `timestamp,smp`(KST 시간 시작, 원/kWh) CSV로 정규화한 뒤 `paths.smp`를 지정한다. 미입력은 피크 목적함수와 비용 결측이다. 과거 SMP는 사후 가격 평가이며 사전 이용 가능 시점은 별도 확인한다.
7. 현재 신청에서 제외한 KEP_007은 현장 템플릿의 `null`을 유지한다. 해당 선택 단계 생략은 의도한 부분완료이며, mock 전체 실행은 기존 경로 회귀 검증을 위해 합성 KEP_007을 포함한다.
8. **반입 묶음:** 코드 zip과 데이터 CSV만 가능하며 최대 10개·총 50MB다. 계산 예: 코드 zip 1 + 데이터 CSV 최대 6(법정동코드 마스터·읍면동 경계 도형·SMP·충전소 위치·행정동별 EV 등록·(선택)공동주택 단지정보) + 폰트 1 = **8개**. V9 0-5절은 데이터 6개만 세므로 코드 zip과 폰트를 더해 세어야 한다. `python tools/check_import_bundle.py <폴더>`로 개수·용량·형식을 점검한다.
9. RAM이 8GB 이하이면 `params.shc.sido`(시도 코드 앞 2자리 목록)와 `chunksize`를 줄여 SHC를 시도별로 나눠 읽는다(`kepco.sido`와 같은 취지, 청크 단위 필터).

산출물: `out_dir/png/`(반출용) · `out_dir/csv/`(현장 작업용, 반출 대상 아님) · `out_dir/pipeline.log`.

## 참고자료 생성

법정동 마스터·코드대응·중심점 참고자료는 `data/ref/` 원본에서 `tools/build_reference_files.py`로 필요할 때 생성한다. 이전 `dist/` 반입 ZIP은 V9 코드와 일치하지 않아 보관하지 않는다.

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
| 4.5 MDE | `identification.py` | `s45_mde` — 실제 처치 수 가정의 주·위약·할인 후 경험 SE와 MDE | 격리 |
| 5 이벤트 스터디 | `identification.py` | `s5_event_main` · `s5_pretrend_*` · `s5_event_regression` · `s5_s6_event_study` | 중단 |
| 6 위약 | `identification.py` | `s6_event_placebo` · `s6_discount_by_k` · `s6_post_summary` | 중단 |
| 6.5 처치오염 | `diagnostics.py` | `s65_contamination` | 격리 |
| 7 CATE | `heterogeneity.py` | `s7_features` · `s7_cate_region` · `s7_cate_validation` · `s7_subgroup_cells`/`s7_importance` | 격리 |
| 8 처방 | `load_axis.py` | `s8_load_concentration` · `s8_quadrants` · `s8_quadrants_plot` | 격리 |
| 8-A 일별 예측 | `load_forecast.py` · `kepco_loader.py` | `s8a_validation` | 격리·실패 표시, 실측 대체 금지 |
| 8-B ESS | `ess_optimizer.py` | `s8b_scenarios` · `s8b_schedules` · `s8b_comparison` · `s9_public_review` | 격리·행별 제약/예측/가격 상태 기록 |
| 8-C 접근성 | `equity_access.py` | `s8c_accessibility`(2SFCA + 지표 1 `ev_per_charger` + 지표 2 `nearest_charger_km`) | 격리·외부자료 없으면 생략. 지표 1·2는 **법정동 중심점 근사**다: 충전소를 가장 가까운 중심점에 배정하며 폴리곤 공간조인·격자점이 아니다(행정경계 근처 충전소는 이웃 동네로 배정될 수 있음). **지표 3(자가충전 제약 주거 비율)은 미구현 — LH KLH_001이 신청 범위에 없어 신청 범위 결정 필요.** 300/500m 커버리지 비율은 8-D 후속(미구현) |
| 9-H 발표 3숫자 | `headline.py` | `s9_headline` | 격리·8-B 또는 8-C 결과 필요. 숫자 1 = 충전기 1기당 EV 수의 상위10%÷하위10% 배수, 숫자 2 = ESS 미적용 평균 상한 초과 kWh, 숫자 3 = 잔여 초과 kWh와 SMP 기준 비용 차이. 각 행에 "증설 N대·상한 배율·이용률 배율" 가정과 상한 배율 범위를 병기. PNG는 소표본 법정동 기여분 제외 |
| 8-E 결합점수 | `priority_score.py` | `s8e_priority`(순위 대상만) · `s8e_not_ranked`(부하 미평가 — 접근성 부족만 확인) · `s8e_priority_summary`(풀 크기·안전/경제성 축 상관) | 격리·8-B/8-C 없으면 생략. 세 축이 모두 있는 동네만 순위·`robust_top`·킬 15번 분모에 든다(`params.priority.min_axes`, 기본 3). 결측 사유는 `not_assessed_low_concentration`(8-B 대상 아님)과 `data_missing`. 안전·경제성 상관 ≥ `params.priority.redundant_corr`(기본 0.9)이면 실행 요약에 "사실상 두 축" 경고 |
| 9 반출 | `outputs.py` · `pipeline.py` | `s0_run_summary` · `s9_manifest` | — |
| 킬 크라이테리아 | `kill_criteria.py` | `kill_criteria/png/k_kill_criteria` | 항목별 `오류`. 16번 = 원천 수록 기간(활성화 창 끝과 비교) |

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
| 분석 지역 단위 | `params.analysis_level`: `"emd"`(법정동, 기본) · `"sigungu"`(시군구 폴백) | 한전 읍면동이 행정동이어서 법정동과 못 맞출 때(킬 5번 실패) `"sigungu"`로 다시 실행한다. 지역키는 시군구5자리+`00000`의 10자리 `bjd_code`이고, 한전은 시도+시군구로만 매칭하며 신한카드·EV는 코드를 시군구로 접는다. 읍면동 시간별 값은 시군구로 합산하고 소표본 억제용 고객호수도 합(공개 값을 만든 인원)으로 잡는다. **300/500/800m 2SFCA와 중심점 근사는 시군구 규모에서 의미가 없어 8-C·8-E는 생략**되고 8-B(안전·경제성)와 9-H 숫자 2·3은 나온다. 지역이 적으면(수십 곳 미만) MDE(4.5)·처치/대조 표본 부족으로 상권 단계가 생략될 수 있다. 해상도가 낮아진 결과이므로 "시군구 단위 잠정 검토"로만 보고한다. 행정동→법정동 배분표 방식(②)은 미구현 |
| 001/002 | 합산하지 않음. 활성화 시점은 `kepco.source`(제공 설정 002), 전력축은 001. **주의: `config.py` 기본값은 `kepco.source="001"`이고 `field_template.json`·`mock.json`은 `"002"`다(통일 여부는 팀 확인 필요).** 산출물 제목은 `activation.source_label`(기본: 002 "공용(사업자 채널) 충전 활성화", 001 "충전 활성화(전체)"). KEPCO_002는 가공일자(20250825)가 제공기간 끝(20251231)보다 이르므로 킬 크라이테리아 16번이 원천별 마지막 수록 달을 창 끝과 비교해 사후 `ratio_months`를 못 채우는 달 수를 적고, `activation.clip_to_data=true`면 창 끝을 마지막 수록 달로 줄인다 | 실제 포괄 범위·사업자 채널 대응은 현장 명세 확인 |

## 공학 산출물 해석

- 상한은 평가 이전 28일 기존 충전 부하 최대값의 1.1/1.2/1.3배다. 변압기 용량 추정값이 아니다. 증설은 0/2/3/5대, 이용률은 CAN 상대 모양 또는 설정 모양에 가정 배율을 적용한다.
- ESS는 이산 후보에서 초기·종단 에너지, SOC 사용 범위, 충방전 효율·출력, 재충전, 순부하 0 이상을 검증해 선택한다. 전역 최소 비용 용량이 아니다. 하루씩 독립된 실험이며 열화·설비비를 제외한다.
- forecast와 oracle에 같은 사전 선정 용량을 사용한다. 예측 오차로 실측 적용 시 상한·종단잔량을 못 지키면 그대로 보고한다. SMP 비용 차이는 실제 전기요금·교체비 절감이 아니다.
- 공공성 증빙 미입력은 “자료 보완 후 검토”다. 낮은 필요도로 점수화하지 않는다. 부하 점검 순서는 미적용 초과 kWh → 잔여 초과 kWh 내림차순, 동률은 법정동코드다. 현장 증빙·비용효과 없이 투자 확정 순위로 쓰지 않는다.
- V9 형평성 점수는 2SFCA를 반전해 접근성이 낮을수록 높다. 결측 축은 0점 대신 남은 가중치로 재정규화하고 신뢰도·결측 축 수를 별도 표시한다. 기본 증설 시나리오는 3대이며 설정으로 바꿀 수 있다.
- 경제성 점수는 SMP 기반 ESS 운영비 절감 잠재력이다. 설치비·배전망 보강비·생애주기비가 없어 공공 투자 순편익으로 해석하지 않는다. MCLP·실제 GIS 지도·DEM 경사 보정은 현재 구현 범위가 아니다.
- 코드 마스터는 도형이 아니다. 실제 2D 지도는 [국토교통부 경계 WFS](https://www.data.go.kr/data/15059008/openapi.do) 등의 기준일·좌표계·코드 정합을 확인한 후 QGIS에서 작성한다. 현재 Python 결과는 표·차트이며 지도 완성을 주장하지 않는다.
- 충전기 시간 이동 최적화는 도착·출차·충전 필요량 등 서비스 제약을 확보한 뒤 수행할 후속 기능이다.

## 반출 규칙 (코드로 강제)

- 모든 표는 CSV + PNG 동시 생성. PNG 가 반출용.
- `paths.font`가 있으면 해당 폰트를 등록하며, 실행 요약에 실제 폰트명을 남긴다. 한글 폰트가 없으면 `params.outputs.font_fallback`이 정한다: `"ascii"`(기본)는 PNG의 한글을 영문 라벨로 바꿔 계속 진행하고(법정동은 `bjd_code`, 사전에 없는 한글은 `?`; 실행 요약 `사용폰트` 열과 킬 크라이테리아 11번이 "영문 대체 라벨로 진행"으로 표시), `"none"`이면 킬 크라이테리아 실패로 본다.
- 위도·경도·좌표 계열 컬럼이 든 표는 `OutputWriter` 가 저장을 거부(`ValueError`). CAN·KEP_007 좌표는 내부 계산에만 쓴다.
- CAN 법정동 표는 차량/세션 수 3 미만 칸을 `—` 로 억제.
- KEPCO 법정동 표는 **공개하는 값을 만든 셀의 고객호수**가 3 미만이면 PNG 수치를 `—`로 억제하고, 내부 CSV에는 원값과 `소표본억제` 열을 남긴다. 법정동 단위 기간 집계 표(s8·8-A·8-B·9·8-E)는 그 기간 시간별 고객호수의 **최댓값**(`params.kepco.suppress_basis`, 기본 `"max"`, 서로 다른 시각은 서로 다른 하한이라 하나의 저조한 시간대로 전체를 가리지 않기 위함)을, 법정동×월 단위 표(s1)는 그 달의 대표 고객호수를 기준으로 한다. `bjd_code` 열이 없는 전국 합계표(s1_kepco_monthly·s1_kepco_band_share)는 행을 가리는 대신 소표본 법정동의 기여분 자체를 PNG 집계에서 빼 차감 역산을 막고, 활성화 시점 예시 그림(s2_activation_examples)도 소표본 법정동을 후보에서 뺀다. 고객호수 정보가 없는 법정동은 보수적으로 계속 억제한다.

소표본 기준 값은 `params.output.min_cell_count`(기본 3, 옛 이름 `params.can.min_cell_count`는 별칭 — 둘 다 있으면 `output`이 우선)이고, 실행 요약(`s0_run_summary`)에 실제 적용된 `min_cell_count`·`suppress_basis`가 남는다. **`suppress_basis`는 KEPCO 법정동 표에만 적용되며 CAN 표에는 적용되지 않는다.** 어느 값을 쓸지는 아래 표로 정한다(사무국 답변 후).

| 고객호수 정의 | 센터 억제 규칙 | suppress_basis |
|---|---|---|
| 고정 등록 수 | 무관 | max (min과 같음) |
| 시간마다 바뀜 | 최종 숫자를 만든 인원만 기준 이상이면 됨 | max |
| 시간마다 바뀜 | 집계에 들어간 칸이 하나라도 기준 미만이면 안 됨 | min |

- 경로 B 산출물 제목에는 "정황상 보조 근거, 개별 차량 식별 아님"을 붙인다.
- CATE 피처에 부하·집중도·피크 계열 이름이 들어오면 `ValueError`, 스냅샷이 활성화 창 시작 이후 달을 포함하면 `ValueError`.

## mock 검증 결과 (9/15, seed 42)

| 확인 | 심은 값 | 결과 |
|---|---|---|
| 법정동 매칭 | 31곳 중 미매칭 1(행복동), '제1동' 표기 6, 동명이인 '중앙동' | 원문 24 · 정규화 6 · 미매칭 1 (3.2%) |
| 활성화 시점 | 처치 18 · never 10 · 창 이전 1 · 창 이후 1 | 18/18 월 단위 일치, 상태 분류 전부 일치 |
| 주 효과(사후 평균) | 평균 τ 0.117 | CS 0.106 (SE 0.036) · 회귀 0.110 (SE 0.032) · 사전추세 p=0.26 |
| 위약 | 0 | 0.023 (SE 0.021) → 할인 후 0.083 |
| MDE 50회 | 실제 처치 18 · 대조 10 | 주 0.0418 · 위약 0.0760 · 할인 후 0.0914, 반복당 0.106초 |
| MDE 고잡음 | SHC 셀 로그정규 잡음 0.10 | 할인 후 경험 SE 0.05110 · MDE 0.14309 |
| 8-A 메모리 | 동일 mock 시간별 파일 | tracemalloc 27.89→22.08 MiB(20.8% 감소), 검증 15행 동일 |
| 처치오염 | 2곳 | 정확히 2곳 유보 |
| KEPCO 소표본 억제(9/19, R1 최솟값→최댓값 기준 변경 전후) | s8_load_concentration(30행)·s8_quadrants(28행)·s8a_validation(15행)·s8b_scenarios(2520행)·s8b_schedules(60480행)·s9_public_review(15행)·s8e_priority(30행) | 억제 건수 변경 전→후: 24→0 · 24→0 · 15→8 · 2520→1344 · 60480→32256 · 15→8 · 30→21. 추가로 s1_kepco_monthly·s1_kepco_band_share 전국 합계 PNG에서 법정동×월 445칸(밴드 표는 2670칸) 제외, s2_activation_examples 예시에서 소표본 법정동 2곳 제외(R2) |
| 법정동별 사전추세 | 0곳 | 유보 0/18 (합성 패널에 심은 추세 1곳은 검출 — 테스트) |
| CAN | individual / model 파일 | individual → 경로 A, model → 경로 B |
| 킬 크라이테리아 | 후보 18곳 · 002⊆001 | 15항목: 통과 11 · 경고 4 · 실패/오류 0 |
| 통합 테스트 | — | 55 passed · 2 skipped · pyflakes 0 |

4사분면 유보 12곳 중 10곳은 'CAN 거점성 충전 다수'다 — mock 에서 자가충전 거점을 12개 법정동에 몰아 둔 탓이며, 실데이터에서 이 비율이 높으면 `can.base_share_flag` 기준을 재검토할 것.

9/15에는 Causal Forest가 미검증이었으나, 9/17에 별도 Python 3.10·EconML 0.17.0 환경에서 120개 합성 지역(처치 60개)의 실제 학습·교차검증·효과 추정 경로를 검증했다. 기본 30지역 mock은 처치 표본 문턱 미달로 여전히 2×2 폴백이 정상이다. 실제 자료에서 Causal Forest의 성능이 검증됐다는 뜻은 아니다. 현장에서는 `s7_cate_validation`을 확인한다.
