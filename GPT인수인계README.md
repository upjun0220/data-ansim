# GPT 인수인계 README — 충전 부하, 하루 먼저 본다 (data-ansim v11)

> **쓰는 법:** 이 문서 전체를 GPT 대화 첫 메시지에 붙여 넣고 "이 프로젝트를 이어서 진행한다"고 말한다.
> 코드를 고치게 할 때는 해당 `.py` 파일 원문도 함께 붙여 넣는다(GPT는 저장소를 직접 읽지 못할 수 있다).
> 작성 2026-09-21 · 기준 커밋 `fdb6355`(GitHub `upjun0220/data-ansim`, `main`·`v11.3` 브랜치 동일)

---

## 1. 프로젝트 한눈에

- **대회:** 2026 데이터+AI 혁신 챌린지(데이터안심구역 부문). 원자료는 센터(인터넷 없는 보안 PC) 안에서만 보고, 동네 단위로 묶은 표·그림만 심사를 거쳐 반출한다.
- **주제:** 서울 법정동마다 **다음날 충전 부하를 AI로 확률 예측(P50·P90)**해 "평소 최대 대비 급증 위험"이 큰 동네를 전날 찾고, **공용 충전 접근성이 부족한 동네**와 합쳐 공공 충전 인프라 투자 검토 우선순위 지도를 만든다. 확장안으로 2028·2030 충전 부하 시나리오.
- **핵심 질문:** 충전 접근성이 부족한 동네 가운데, 내일 저녁 충전 부하가 평소 최대를 넘어 급증할 위험이 큰 곳은 어디인가?
- **분석 지역:** 서울 법정동만(법정동코드 앞 2자리 `11`). 형평성 분모인 동 단위 전기차 등록 자료가 서울뿐이라서.
- **DSZ 신청 데이터 4개:** KEPCO_001(법정동·시간별 충전량), KEPCO_002(사업자 채널), 티벨 CAN(M-Type), LX DEM5M. 신한카드 SHC001·002는 v10.1에서 제외.

### 정직한 스코핑(절대 원칙)
- 급증 위험은 **변압기 과부하가 아니다.** "그 동네 교정기간 최대 부하 × 1.1~1.3배를 넘는 급증"이다. 상한은 정책 가정.
- 정전 예방·교체비 절감은 **입증한 성과가 아니다.** ESS 용량은 이산 후보 중 첫 실행가능 후보이지 최적이 아니다.
- AI는 **그래디언트 부스팅 분위수 회귀**다. 딥러닝·실시간 제어라고 부르지 않는다.
- 2030은 **시나리오(예측 아님), 충전기 추가 설치 없음 가정.**
- **mock(가상) 데이터에서 AI가 이겨도 성능 근거가 아니다**(기온·요일·휴일 반응을 직접 심었음).

---

## 2. 현재 상태

| 항목 | 상태 |
|---|---|
| 코드 | v11.3 설계 전부 구현 완료. 테스트 **113개 통과·3개 생략**(econml·ruptures·lightgbm 미설치 경로) |
| 저장소 | GitHub `upjun0220/data-ansim` — `main` = `v11.3` = `fdb6355` |
| 로컬 작업 폴더 | `C:\Users\verty\OneDrive\바탕 화면\dataansim-v11.3` |
| 설계 문서 | `docs/1_팀공유_전체설계_V11.4_20260921.html` (V11.3 설계 + 구현 확정 사항 + 8절 구현 결과) |
| 팀 안내서 | `0_로드맵북_센터가기전_필독_v11.html` (코드 파일별 역할·작동법·센터 절차) |
| 반입 폴더 | 코드 파일을 따로 반입(모듈 20개 + `fieldtemplate.txt`·`holidays.txt`·`requirements.txt`·`README.txt`·`checksums.txt`) + 데이터 `5.csv`·`9.csv`. 만들기 `python tools/make_import_bundle.py [폴더]`. 로컬 사본 `다운로드\반입v11\upload` |
| 삭제한 것 | V9·V10 설계, 로드맵북 v10, V11.3 설계, 연구계획서 0917·0919(git 기록엔 남음), 이전 반입 폴더(휴지통) |

---

## 3. 파이프라인 (실행 순서)

```
0·1 정합·집계 → 1-C KEPCO_002 채널 비교 → 3 CAN u(t) → [2·4~7 상권: 기본 비활성]
→ 8 부하 집중도 → 8-C 접근성 → 0-E 공휴일·기온 점검 → 8-A 기준 모델 검증 → 8-A AI 분위수·급증 위험
→ 8-B ESS·AI 효과 → 8-E 결합 점수 → 8-F 시나리오 → 9-H 발표 숫자 → 9 반출(요약·목록)
```
- **0·1만 필수**, 나머지는 격리(실패해도 이유를 실행 요약 `비고`에 남기고 계속). 9단계 요약·목록은 실패해도 반드시 남긴다.
- 폴백: lightgbm 없으면 sklearn `HistGradientBoostingRegressor(loss="quantile")`, 그것도 없으면 기준 모델("AI 예측 미실행"). scipy 없으면 ESS 그리디. SMP 없으면 비용 차이 결측. 8-C 실패 시 8-A는 동네 특성 없이 학습.

## 4. 모듈 (파일명에 밑줄 없음 = 반입 포털 규칙: 영문·숫자만)

| 파일 | 단계 | 역할 · 주요 함수 |
|---|---|---|
| `pipeline.py` | 전체 | 단계 연결·격리·실행 요약. `run(config_path)`, `channel_share()` |
| `killcriteria.py` | 첫날 점검 | 21항목 판정. `run_checks(config_path)` |
| `config.py` | 설정 | `DEFAULT_PARAMS`, `load_config()`, `resolve_region()` |
| `common.py` | 공구 | CSV 읽기(인코딩·구분자 자동), `keep_region()`, 거리·p값 |
| `bjdmapping.py` | 0 | 한전 동네 이름 ↔ 법정동 코드, 코드대응 |
| `kepcoloader.py` | 1 | KEPCO 청크 읽기·집계, `load_kepco_hourly()`(8-A 입력) |
| `weatherloader.py` | 0-E | 기온(`forecast_temps`, `assert_forecast_before_cutoff`), 공휴일(`load_holidays`, `classify_holidays`) |
| `canloader.py` | 3 | CAN 세션·u(t)·반복 위치. 상권 비활성이면 처치 정제 안 함 |
| `loadaxis.py` | 8 | 집중도·`classify_load()`(4사분면은 상권 전용) |
| `equityaccess.py` | 8-C | 2SFCA, 지표 1·2, `ev_counts` 인자로 이력 기반 대수 입력 |
| `loadforecast.py` | 8-A | `forecast_baseline`, `build_features`, `assert_no_leakage`, `select_model`, `fit_quantiles`, `run_ai_forecast`, `evaluate_forecasts`, `surge_risk`, `risk_summary`, `ai_improvement` |
| `essoptimizer.py` | 8-B | LP/그리디, `run_scenarios(..., ai_pred)`, `ai_effect`, `ai_effect_summary`, 공용 `policy_target`·`utilization_profile`·`compute_added_load` |
| `priorityscore.py` | 8-E | `build_priority_v11`, `choose_risk_metric`, `run_sensitivity_v11`, 레거시 `build_priority`(세 축·AHP) |
| `loadscenario.py` | 8-F | `load_ev_history`, `load_hdong_bjd`, `map_to_bjd`, `growth_rate`, `solve_k_goal`, `estimate_beta`, `typical_curves`, `run_scenarios_8f`, `summarize_scenarios` |
| `headline.py` | 9-H | `headline_table_v11`(v11 숫자), 레거시 `headline_table` |
| `outputs.py` | 9 | CSV+PNG 저장, 좌표 열 거부, 소표본 억제, 폰트 대체 |
| `identification.py`·`diagnostics.py`·`heterogeneity.py`·`kep007loader.py` | 상권 | 보존만(`stages.commerce=true`일 때만) |
| `mock_data.py` | 연습 | `--energy`(기온·공휴일·이력·대응표), `--with-shc`(상권 회귀용) |
| `tools/fetch_holidays.py` | 준비 | 특일 정보 API → `config/holidays.yaml`(서비스키 환경변수 `DATAGOKR_SERVICE_KEY`) |
| `tools/make_import_bundle.py` | 반입 | 코드를 파일별로 반입 폴더에 담고 설정·달력·README를 .txt로, `checksums.txt` 작성, 새 폴더에서 import·설정 읽기 검증 |
| `tools/check_import_bundle.py` | 반입 | 형식·개수·용량·파일명·이전 반입 이름 중복 점검 |

## 5. 핵심 정의 (코드와 일치)

- **기준 모델:** 같은 요일·시간 최근 4주 중앙값. 공휴일은 같은 유형 과거 휴일 4개, 부족하면 실패(임의 대체 금지).
- **AI:** 서울 법정동 전체를 한 모델로(global, 동 ID 미사용). 특성 = 시간·요일·휴일유형·전날 같은 시간·7일 전·기준 모델 값·기온 예보·동네 특성(전기차 수·충전기 수·**학습기간 평균 부하**). P50·P90 따로 학습, `q90 = max(q90, q50)`. 격자 8조합(학습률{0.05,0.1}×잎{15,31}×최소 잎 표본{20,60}), 확장창 3겹 pinball.
- **기간:** 학습(검증 시작 전 182일) < 검증(평가 시작 전 8주) < 평가(**28일**). 기본 평가 시작 `2025-12-04`.
- **누설 방지:** 예측일 d의 입력은 d−1일까지 관측, 예보는 발표 시각 ≤ d−1일 18:00 중 최신. 어기면 예외(킬 19).
- **급증 위험(점수용, 증설 0대):** 경보 = `max_h q90 > 교정기간 최대 × 배율`, `surge_risk` = 평가일 중 경보 비율, `peak_ratio` = 평균 max q90 ÷ 교정기간 최대. 정밀도·재현율은 AI·기준 모델이 모두 있는 **공통일**만(제외 일수 `n_excluded_days`). 교정기간 최대 < 1kW면 `not_assessed_low_load`. `surge_risk_with_new_N`(증설 가정)은 참고 레이어, 점수 제외.
- **AI 효과:** 증설 3대 × 상한 1.1·1.2·1.3에서 같은 용량으로 기준 모델·P50·P90·oracle 운전. `1 − 초과kWh(P90)/초과kWh(기준)`, 공통일 합산, 기준 초과 0이면 대상 제외. 과잉 방전 kWh 병기. 음수도 그대로.
- **8-E:** `0.5·위험_norm + 0.5·형평성_norm`, `min_axes=2`, 결측 축은 0점이 아니라 재정규화, 순위 대상의 50% 이상이 급증위험 0이면 `peak_ratio`로 자동 대체(킬 21). 민감도 = 상한 3배율 × w_risk 0.35~0.65(7단계). 두 축 피어슨·스피어만 상관 보고.
- **8-F:** `EV_Y = EV_now × (1 + k·g)^(Y−Y0)`, g = 36개월 연평균 증가율, 기준월 2026-06(→2030년 말 4.5년). 저 k=0.5, 중 1.0, 고 `k_goal`(Σ EV_2030 / Σ EV_now = 4,200,000/1,095,218 ≈ 3.835, 이분법). β = log(대표 부하)~log(전기차) OLS + 부트스트랩 95%; 0 포함 또는 [0.3, 2.0] 밖이면 β=1(0.8·1.2 민감도). `L_Y = L_typ × (EV_Y/EV_now)^β` — **트리 모델로 외삽 금지.** 행정동→법정동은 **대응표(가중치)**로 배분(팀 결정 (b)안).
- **9-H 숫자:** 1 = 충전기당 전기차 상·하위 10% 배수 / 2 = 증설 0대 경보 정밀도·재현율(AI=`값`, 기준=`비교`) → AI 효과 평균·범위·대상 수 / 3(a) = 급증위험 ≥ 0.1인 동네 수, 3(b) = 2030 중 시나리오 1.2배 상한 초과 동네 수(`비교` = 고 시나리오). AI 미실행·킬 18 실패면 "미실행"/"AI 개선 없음" 표기.

## 6. 주요 설정 키 (`config/field.json`, 없으면 `config.py` 기본값)

| 키 | 기본 |
|---|---|
| `stages.commerce` | `false` |
| `region.sido_prefix` / `sido_name` | `"11"` / `"서울특별시"` (`kepco.sido`보다 우선) |
| `weather.mode` / `fcst_cutoff` | `"forecast"` / `"18:00"` |
| `holidays.long_min_days` / `count_weekends` | `3` / `true` |
| `energy.evaluation_start` / `evaluation_days` | `"2025-12-04"` / `28` (현장 템플릿은 `null` — 반드시 입력) |
| `forecast.model` / `train_days` / `min_p90_coverage` | `"auto"` / `182` / `0.8` |
| `risk.min_calib_peak_kw` | `1.0` |
| `ess.forecast_input` / `compare_inputs` / `compare_new_chargers` | `"p90"` / `true` / `3` |
| `priority.axes` / `weights` / `min_axes` / `risk_metric` | `["risk","equity"]` / `{"risk":0.5,"equity":0.5}` / `2` / `"auto"` |
| `headline.risk_threshold` | `0.1` |
| `scenario.base_month` / `goal_national` / `national_base` / `seoul_base` | `"2026-06"` / `4200000` / `1095218` / `118967`(점검용) |
| `output.min_cell_count` / `kepco.suppress_basis` | `3` / `"max"` |
| 경로 | `weather`, `holidays`, `ev_history`, `hdong_bjd` 신규. `ev_registration`이 비면 8-C도 이력 기준월 값 사용 |

## 7. 반입·반출 규칙

**반입(포털 규칙):** `.py`·`.csv`·`.txt`·폰트만(zip·json·md 불가), 파일명 영문·숫자만, 총 50MB(파일 개수 제한 없음). **이전 반입 이름(`dataansimbundle.py`, `dataansimcodev10.zip`, `bjdmaster.csv`, `bjdcrosswalk.csv`, `bjdcentroids.csv`, `smp.csv`, `evstations.csv`, `evregistration.csv`, `koreanfont.ttf`)과 겹치면 안 된다.**

| 이름 | 내용 |
|---|---|
| 모듈 20개 `*.py` + `fieldtemplate.txt`·`holidays.txt`·`requirements.txt`·`README.txt`·`checksums.txt` | 코드·설정(내용은 JSON, 확장자만 .txt). 센터에서 한 폴더에 평평하게 둠 |
| `2.csv`·`3.csv`·`4.csv` | 법정동코드 마스터·코드대응·중심점 |
| `5.csv`·`6.csv` | SMP(`timestamp,smp`)·공용 충전소 위치 |
| `7.csv`·`8.csv` | 행정동별 전기차 등록 월별 이력(OA-21236)·행정동→법정동 대응표(가중치) |
| `9.csv` | 기온 `timestamp,station_or_grid,temp_obs_c,temp_fcst_c,fcst_issued_at` |
| `10.ttf` | 한글 폰트(없을 때만) |

공휴일 달력 `config/holidays.yaml`은 번들 안에 들어간다(JSON 문법 = 유효한 YAML, PyYAML 불필요). 2024~2025 공휴일 38일(대체·임시·선거일 포함, Nager.Date·정부 발표 대조)이 들어 있다.

**반출(코드로 강제, 약하게 만들지 말 것):** 위도·경도·좌표 열이 든 표는 저장 거부(`OutputWriter`). 법정동 행은 대표 고객호수 < `min_cell_count`면 PNG에서 `—`. 법정동 열이 없는 합계·요약표는 소표본 법정동을 **빼고 다시 계산**해 PNG에 싣는다(차감 역산 방지). 새 표를 만들 때도 같은 규칙. **`png_df=None`을 넘기면 원본이 그대로 PNG로 나가므로, 억제 후 비면 빈 표를 넘긴다.**

## 8. 킬 크라이테리아 (코드 번호)

1 패키지 · 2 T_r 분포(상권 off면 해당 없음) · 3·4·6 SHC(해당 없음) · 5 한전 매핑 실패율(30%↑ 실패 → `analysis_level="sigungu"`) · 7 002⊆001 · 8 CAN 식별번호 · 9 scipy · 10 MDE(해당 없음) · 11 한글 폰트 · 12 접근성 매칭 · 13 계절 커버리지 · 14 가중치 유효성(합 1) · 15 강건 상위군(→ 축별 순위표 2장) · 16 002↔001 대조 가능 기간 · 17 AI 라이브러리 · 18 AI 성능(P50 MAE ≤ 기준, P90 적중률 ≥ 80%, 파이프라인 뒤 판정) · 19 기온 예보·발표 시각 · 20 시나리오 매핑률·β·k_goal · 21 급증위험 0 비율 ≥ 50%. 판정 = 통과·경고·실패·오류·해당 없음. 설계 문서 번호 대응: 설계 11↔코드 2, 설계 10↔코드 8, 나머지 동일.

## 9. 실행 명령 (로컬, 저장소 루트)

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe mock_data.py --out data/mock --energy
.venv\Scripts\python.exe killcriteria.py --config config/mock.json
.venv\Scripts\python.exe pipeline.py --config config/mock.json
.venv\Scripts\python.exe killcriteria.py --config config/mock.json   # 18·21번은 분석 뒤 판정
.venv\Scripts\python.exe -m pytest -q tests                         # 113 passed, 3 skipped 기대
.venv\Scripts\python.exe tools/make_import_bundle.py                # dist/import 에 파일별 반입 폴더
.venv\Scripts\python.exe tools/check_import_bundle.py dist/import
```
센터: 반입 파일을 모두 한 폴더(예: `dataansim11`)에 두고 `%cd` → `fieldtemplate.txt`를 `field.txt`로 복사·KEPCO·CAN 경로와 `evaluation_start` 입력 → `from killcriteria import run_checks; run_checks("field.txt")` → `from pipeline import run; run("field.txt")`. 해시 확인은 `checksums.txt`.

## 10. mock 결과 (seed 42, 동작 확인용 — 성능 근거 아님)

- SHC 없이 전 단계 완료, 킬 21항목 오류 0.
- 8-A(sklearn, 기온 예보, 검증 56일): MAE persistence 0.542 · 기준 0.486 · AI 0.433 / 피크 시간 오차 기준 1.487 · AI 1.496 / **P90 적중률 0.77 → 킬 18 실패("AI 개선 없음")** / 27/30곳 AI 우세.
- 급증위험 0인 동네 93% → 위험 축 피크비율 대체(킬 21 경고). 12-25(단일 공휴일)는 기준 모델 실패로 비교 제외.
- AI 효과(증설 3대): 1.2배 0.21(대상 10/15), 1.1배 0.32, 1.3배 0.70(대상 2). P90 과잉 방전이 더 큼(81.8 대 9.3 kWh).
- 8-E 순위 대상 30/30곳(V10은 절반 미평가). 8-F: 매핑 실패 0.35%, β 불안정 → 1, k_goal 1.22.

## 11. 남은 일 · 팀 결정 대기

1. **데이터 준비:** 완료 = `5.csv` SMP(EPSIS)·`9.csv` 기온(Open-Meteo: KMA 예보모델 48시간 전 값·ERA5 실측 — 기상청 ASOS 원자료 아님)·공휴일 달력(`tools/fetch_public_inputs.py`). 남음 = `2~4.csv`(원본 `data/ref/` 없음), `6.csv` 충전소, `7.csv` 등록 이력, `8.csv` 대응표.
2. **서류:** 연구계획서·DSZ 신청서의 분석 지역 "수도권→서울", 반입 목록 1~10번으로 수정.
3. **결정:** 숫자 3 헤드라인 문구·`risk_threshold`, KEPCO_002 유지 여부, 경계 도형·공동주택 자료 추가 반입 여부, 기준월 2026-08 이동 여부(`national_base`·`seoul_base` 통계누리 대조).
4. **센터 문의:** lightgbm·sklearn 설치 여부, 한전 읍면동이 법정동인지, 고객호수 정의·소수 셀 기준, KEPCO_002 실제 기간, 메모리·폰트.
5. **미구현(후속):** 지표 3(자가충전 제약 주거 비율), 300/500m 커버리지(8-D), MCLP, DEM 경사 보정, 실제 GIS 지도(안심구역 태블로로 CSV와 경계 결합하는 경로 검토 — 반출 그림에도 억제·좌표 규칙 적용).

## 12. GPT가 지킬 작업 규칙

- 코드를 고치면 `pytest -q tests`와 mock 3단계(데이터 생성 → 킬 → 파이프라인)가 끝까지 도는지 확인한다. 기존 테스트는 지우지 않는다.
- 단계 **격리 설계**를 깨지 않는다. 0·1만 필수.
- **KEPCO_001과 002를 합산하지 않는다.**
- **휴일을 규칙으로 생성하지 않는다**(날짜는 파일·API·관보에서만).
- **실측 기온 모델 결과를 발표 숫자·경보에 쓰지 않는다.** 예보는 전날 18시 이전 발표분만.
- **트리 모델로 시나리오를 외삽하지 않는다.** **딥러닝을 추가하지 않는다.**
- **상권 단계 코드를 삭제하지 않는다**(끄기만).
- 좌표 비반출·소표본 억제 규칙을 약하게 만들지 않고, 새 표에도 적용한다.
- 새 설정 키는 `config.py`의 `DEFAULT_PARAMS`와 `config/fieldtemplate.json`(설명은 `_` 접두 키)에 함께 넣고 README 표를 갱신한다.
- 반입되는 새 파일·모듈 이름은 영문·숫자만(밑줄·하이픈·공백·한글 불가), 이전 반입 이름과 중복 금지. 새 모듈은 `tools/make_import_bundle.py`의 `MODULES`에 추가.
- mock 결과를 실제 성능처럼 서술하지 않는다. 결과를 과장하지 않는다.
- 윈도우에서 저장소 경로가 너무 길면 긴 결과 파일 저장이 실패할 수 있다(260자) — 짧은 경로에서 실행.
