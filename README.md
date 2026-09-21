# 충전 부하, 하루 먼저 본다 (구 충전 리플맵) — 분석 파이프라인

2026 데이터+AI 혁신 챌린지(데이터안심구역 부문) 제출용. 서울 법정동의 다음날 충전 부하를 AI로 확률 예측해 **평소 최대 대비 급증 위험**과 **충전 접근성 부족**을 합친 공공 충전 인프라 투자 검토 우선순위 지도를 만든다.

- **설계 기준:** `docs/1_팀공유_전체설계_V11.4_20260921.html`(수식·근거·Q&A)
- **팀 공부·센터 작업 안내:** `0_로드맵북_센터가기전_필독_v11.html`(코드 파일별 역할·작동법·첫날 절차)
- 실측 변압기 정격·주거지 접속관계·취약계층 수혜를 확보한 분석이 아니며, 정전 예방·교체비 절감은 입증한 성과가 아니다. 결합점수는 실제 설치 지점 선정 결과가 아니라 법정동별 현장 검토 순서다.
- 테스트 113개 통과·3개 생략(미설치 선택 패키지). 이전 버전 문서(V9·V10 설계, 로드맵북 v10, V11.3 설계, 연구계획서 0917·0919)는 저장소에서 지웠으며 git 기록에 남아 있다.

> ⚠ `data/mock/` 은 전부 **가상** 데이터다. 로더·파이프라인 동작 검증용이며 결론에 쓰지 않는다.

## v11 — 현재 기준 (`docs/1_팀공유_전체설계_V11.4_20260921.html`)

**현재 설계 기준은 `docs/`의 V11.4 HTML(V11.3 설계 + 구현 반영)이다.** 이 절 아래의 V10 서술(구조표·폴백·설계 결정·mock 검증 결과)은 이력이며, 이 절과
다르면 이 절이 우선한다.

> **2단계 하이브리드 처방.** Day-ahead AI 예측은 단기적으로 공공 충전 DR(출력제어)·이동형 충전차량/ESS 급파를 위한 운영 지표이며, 급증 경보가 누적되는 지역을 2030 시나리오와 결합해 고정형 공공 충전기·ESS의 중장기 투자 우선순위로 잇는다(설계서 서론, Q15).
>
> **형평성 축은 주거지 충전 기본권.** 업무·상업지구의 주간 부하는 안전 축(KEPCO 실측·AI 예측)이 잡는다. 형평성 수요를 등록대수로 산정하는 것은 민간 CPO가 기피하는 다세대·다가구 주민의 야간 자가충전 사각지대를 해소하는 공공 복지 목적이다(07-3절, Q16).
>
> **책임성·신뢰성(Trustworthy AI) 4대 원칙:** 누설 차단 · 이중 안전장치 · 과잉 방전 공개 · 민감도 검증(설계서 5절).

> **정직한 스코핑.** 급증 위험은 변압기 과부하가 아니라 **"그 동네의 교정기간 최대 부하 × 배율을 넘는 급증"**이다. 목표 상한은
> 정책 가정이지 공학 기준·변압기 용량이 아니다. AI는 그래디언트 부스팅 분위수 회귀이며 딥러닝·실시간 제어가 아니다. 8-F는
> **시나리오(예측 아님), 충전기 추가 설치 없음 가정**이다. **mock에서 AI가 기준 모델을 이겨도 성능 근거가 아니다** — 기온·요일·
> 휴일·추세 반응을 mock에 직접 심었기 때문이다.

### 파이프라인 순서

`0·1(정합·집계) → 1-C(KEPCO_002 채널 비교) → 3(CAN u(t)·KEP_007) → [2·4~7 상권 단계: 기본 비활성] → 8(load_axis) → 8-C →
0-E(공휴일·기온 점검) → 8-A(기준 모델 검증) → 8-A(AI 분위수 예측·급증 위험) → 8-B → 8-E → 8-F → 9-H → 9`

8-A는 8-C의 동네 특성(전기차·충전기 수)을 입력으로 쓰므로 8-C 뒤에 돈다. 8-C가 실패하면 8-A는 동네 특성 없이 학습한다.
0·1만 필수이고 나머지는 모두 격리(실패해도 다음 단계와 9단계 반출은 계속)된다.

| 단계 | 파일 | 주요 산출물 | 실패·없을 때 |
|---|---|---|---|
| 0·1 정합·집계 | `bjdmapping.py` · `kepcoloader.py` | `s0_bjd_match_rate` · `s1_kepco_monthly` | 중단 |
| 1-C 채널 비교 | `pipeline.py` | `s1_channel_share`(002/001 비율의 시간대별 분포, **합산하지 않음**) | 002 없으면 생략 |
| 3 CAN | `canloader.py` | 세션 길이·반복 충전 위치 비율·u(t) (처치 정제는 상권 단계에서만) | 격리 |
| 2·4~7 상권 | `identification.py` 외 | `stages.commerce=true` 일 때만 | 기본 비활성 — 실행 요약에 "상권 단계 비활성(v11)" |
| 8 부하 집중도 | `loadaxis.py` | `s8_load_concentration` · `s8_load_type`(고·저, 공공 검토 유형) · 4사분면은 상권 단계에서만 | 격리 |
| 8-C 접근성 | `equityaccess.py` | `s8c_accessibility`(2SFCA·지표 1·2) | 격리 |
| 0-E 외부 입력 | `weatherloader.py` | `s0_external_inputs`(공휴일 달력·기온 지점·예보 사용 가능 시각·기온 모드) | 없으면 공휴일 미보정·기온 none |
| 8-A AI 예측 | `loadforecast.py` | `s8a_validation`(기준 모델) · `s8a_forecast_metrics` · `s8a_risk` · `s8a_risk_summary` | lightgbm → sklearn → 기준 모델 폴백 |
| 8-B ESS | `essoptimizer.py` | `s8b_scenarios`(`forecast_input` 열) · `s8b_ai_effect` · `s8b_ai_effect_summary` · `s9_public_review` | AI 없으면 기준 모델 운전("baseline(AI 없음)") |
| 8-E 결합 점수 | `priorityscore.py` | `s8e_priority` · `s8e_not_ranked` · `s8e_rank_risk` · `s8e_rank_equity` · `s8e_priority_summary` | 8-A·8-C 없으면 생략 |
| 8-F 시나리오 | `loadscenario.py` | `s8f_scenario` · `s8f_scenario_region` · `s8f_beta` · `s8f_scenario_2030_mid`(PNG) | `paths.ev_history` 없으면 생략 |
| 9-H 발표 숫자 | `headline.py` | `s9_headline` | 격리 |

### 8-A — AI 분위수 예측과 급증 위험

- **모델:** 서울 법정동 전체를 한 모델로 학습(global, 법정동 ID 미사용). P50·P90을 따로 학습하고 `q90 = max(q90, q50)`.
  입력: 시간·요일·휴일유형·전날 같은 시간·7일 전 같은 시간·기준 모델 값·기온 예보·동네 특성(전기차 등록, 공용 충전기 수, 학습기간
  평균 부하). 하이퍼파라미터는 학습기간 안 시간 순 확장창 교차검증(pinball)으로만 고른다(격자 8조합).
  - 설계서의 "교정기간 평균 부하"는 검증기간과 겹쳐 누설이 되므로 **학습기간 평균 부하**(`train_mean_kw`)로 바꿨다.
- **기간:** 학습(평가 시작 − 8주 이전 `train_days`일) < 검증(평가 시작 전 8주) < 평가(`energy.evaluation_days`, v11 기본 28일). 겹치지 않는다.
- **정보 누설 방지:** 예측일 d의 입력은 d−1일까지의 관측과, 발표 시각이 d−1일 `weather.fcst_cutoff`(18:00) 이전인 예보 중 최신
  발표분뿐이다. 어기면 `assert_no_leakage`가 예외를 내고 킬 19번이 실패로 표시한다. `observed`(실측 기온)는 상한 참고 모델로만
  학습하며 발표 숫자·경보에 쓰지 않는다. 예보가 없으면 `none`과 `observed`를 함께 학습해 지표만 나란히 낸다.
- **검증(`s8a_forecast_metrics`):** 전체(`bjd_code` 빈칸)와 법정동별 persistence·기준 모델·AI P50 MAE, 증가일 평균 오차,
  피크 시간 오차, P90 적중률, pinball, 검증일 수, 사용 모델, 기온 모드. AI가 기준 모델을 못 이긴 동네는 `ai_beats_baseline=False`로
  남긴다(숨기지 않음).
- **급증 위험(`s8a_risk`, 상한 1.1·1.2·1.3배별, 모든 서울 법정동):** 급증경보(r,d) = 1[max_h q̂0.9 > 교정기간 최대 × 배율],
  `surge_risk` = 평가일 중 경보일 비율(**증설 0대 — 점수용**), `peak_ratio` = 평가일 평균 max_h q̂0.9 ÷ 교정기간 최대.
  실측 급증과 대조한 경보 정밀도·재현율을 AI와 기준 모델에 같은 방식으로 낸다. 두 예측이 모두 있는 **공통일**만 비교하고 제외 일수를
  `n_excluded_days`로 적는다(기준 모델은 같은 유형 휴일 4개가 없으면 그날 예측을 실패로 둔다). 교정기간 최대가
  `risk.min_calib_peak_kw` 미만이면 `not_assessed_low_load`. `surge_risk_with_new_N`(증설 2/3/5대, ΔL 가정)은 **참고 레이어**이며
  점수·검증에 쓰지 않는다. ΔL(`essoptimizer.compute_added_load`·`utilization_profile`)과 목표 상한(`policy_target`)은 8-A·8-B·8-F 공용이다.

### 8-B — AI 입력과 AI 효과

`ess.forecast_input`(기본 p90)으로 스케줄을 짠다. `ess.compare_inputs`면 증설 3대·상한 1.1·1.2·1.3배에서 **같은 사전 선정 용량**으로
기준 모델·P50·P90·oracle을 모두 실측에 재현한다. `ai_effect = 1 − 초과kWh(P90)/초과kWh(기준 모델)`(모든 입력의 계획이 있는 공통일
합산), 기준 모델 초과가 0이면 대상에서 빼고 대상 수를 적는다. 과잉 방전 kWh(실측 기준 필요 없었던 방전)를 함께 낸다. 음수·0 근처도
그대로 보고한다.

### 8-E — 두 축 결합 점수

`PriorityScore = w_risk·위험_norm + w_equity·형평성_norm`(기본 1/2씩). 위험 축은 8-A의 **증설 0대** `surge_risk`이며, `risk_metric=auto`
이면 순위 대상의 50% 이상에서 0일 때 `peak_ratio`로 바꾸고 실행 요약·킬 21번에 남긴다. 형평성은 2SFCA(접근성 낮을수록 높음).
경제성(SMP 비용 차이)·ESS 미적용 초과 kWh·`surge_risk_with_new_N`은 표시 열이다. `min_axes=2`, 결측 사유(`not_assessed_low_load`·
`data_missing`)와 순위 밖 목록 유지, 결측 축은 0점이 아니라 재정규화. 민감도는 상한 3배율 × `w_risk ∈ [0.35, 0.65]`(7단계), 강건
상위군·경계선은 순위 대상 안에서만. 두 축의 피어슨·스피어만 상관을 매번 보고한다. AHP·`axes_redundant`는 레거시 세 축
(`priority.axes=["safety","equity","economy"]`)에서만 동작한다.

### 8-F — 2028·2030 충전 부하 시나리오 (시나리오, 예측 아님)

- 입력: 행정동별 전기차 등록 월별 이력(OA-21236 형식, `paths.ev_history`) + **행정동→법정동 대응표**(`paths.hdong_bjd`, 가중치 =
  그 행정동 전기차 중 법정동 몫; 가중치가 없으면 같은 몫으로 나누고 경고). 매핑 실패율(기준월 전기차 중 대응표에 없는 행정동 몫)을
  킬 20번과 `s8f_beta`에 보고한다.
- `EV_Y = EV_now × (1 + k·g)^(Y−Y0)`, g = 기준월 이전 36개월 연평균 증가율, `Y−Y0`는 기준월 말부터 월 단위(2026-06 → 2030년 말 4.5년).
  저 k=0.5, 중 k=1.0, 고 k=`k_goal`(서울 합계 배율 = `goal_national/national_base` ≈ 3.835가 되도록 이분법). 고 시나리오는 서울의
  전국 비중 유지 가정과 같다.
- β: 동네 간 log(대표 부하) ~ log(전기차) OLS + 부트스트랩 95% 구간. 대표 부하 = 평가기간 평일 저녁 피크 기준 P90 날의 일 최대.
  구간이 0을 포함하거나 `beta_bounds` 밖이면 β=1, 0.8·1.2 민감도 행(`beta_case`)만 낸다. `L_Y = L_typ × (EV_Y/EV_now)^β` —
  **8-A 트리 모델로 외삽하지 않는다.**
- 판정: 현재 상한(1.1·1.2·1.3배) 초과 동네 수, 고 시나리오에서만 추가로 넘는 동네(`over_only_high`), 8-B 후보 격자로 본 필요 ESS
  용량 변화, 지표 1 미래값(충전기 수 현재 유지).
- 기준값: `base_month` 2026-06, `goal_national` 4,200,000(제1차 국가 탄소중립녹색성장 기본계획 2030 목표), `national_base`
  1,095,218(2026-06 국토교통부 자동차 등록현황 연료별 — 6월 말 보도자료 "전기 1,095천대"와 일치 확인), `seoul_base` 118,967(2026-06,
  2차 출처 DataFact, 원자료 미대조 — 계산에 쓰지 않는 점검용, OA-21236 합계와 5% 넘게 다르면 경고). 전국 값은 보도자료로 대조했고
  서울 값은 보도자료에 시도별 수치가 없어 대조하지 못했다(계산에 쓰지 않아 결과에 영향 없음). **통계누리에 2026년 8월 자료까지
  있으므로, 팀이 원자료 xlsx를 받으면 기준월을 2026-08로 옮겨도 된다 — 그때는 세 값과 EV_now(r)의 월을 함께 바꾼다.**

### 9-H — 발표 숫자 (각 행에 증설 대수·상한 배율·이용률 배율·기온 모드·사용 모델 병기)

- 숫자 1: 충전기 1기당 전기차 대수의 지역 간 배수(V10 그대로).
- 숫자 2: 증설 0대 급증 경보 정밀도·재현율(AI = `값`, 기준 모델 = `비교`) → AI 효과 법정동 평균과 범위(상한 1.1~1.3)·대상 수.
  AI 미실행이면 "미실행", 킬 18번 기준 미달이면 "AI 개선 없음(킬 18)"을 가정 열에 적는다.
- 숫자 3: (a) 현재 증설 0대 급증위험 ≥ `headline.risk_threshold`(0.1) 동네 수(상한 1.2배, 평가기간 예측·관측),
  (b) 2030 중 시나리오에서 현재 상한 1.2배를 넘는 동네 수(`비교`·괄호 = 고 시나리오). 둘 다 "충전기 추가 없음" 기준이지만 (a)는 예측·관측,
  (b)는 보급 증가 시나리오다. **헤드라인 문구는 팀이 확정한다.**

### 새 설정 키 (전체는 `config.py`의 `DEFAULT_PARAMS`, 현장 예시는 `config/fieldtemplate.json`)

| 키 | 기본값 | 뜻 |
|---|---|---|
| `stages.commerce` | `false` | `true`면 V10 상권 단계(T_r·Y_ddd·이벤트 스터디·위약·처치오염·CATE·4사분면). 이때만 SHC·업종코드 필수 |
| `region.sido_prefix` / `region.sido_name` | `"11"` / `"서울특별시"` | 코드 앞 2자리 필터 / KEPCO 시도 텍스트 필터(청크 단위). `kepco.sido`는 하위 호환 별칭, region 우선 |
| `weather.mode` / `weather.fcst_cutoff` | `"forecast"` / `"18:00"` | forecast·none·observed(참고용). 예보 없으면 none 폴백 |
| `holidays.long_min_days` / `holidays.count_weekends` | `3` / `true` | 연휴(`long_holiday`) 기준, 앞·뒤 평일은 `pre_post_holiday` |
| `energy.evaluation_start` / `energy.evaluation_days` | `"2025-12-04"` / `28` | 평가 28일(급증위험이 1/7 단위로만 나오지 않게) |
| `forecast.model` | `"auto"` | lightgbm → sklearn(분위수) → baseline |
| `forecast.quantiles` / `train_days` / `cv_folds` / `max_iter` / `grid` / `random_state` | `[0.5,0.9]` / `182` / `3` / `100` / 8조합 / `42` | AI 학습 설정 |
| `forecast.min_p90_coverage` | `0.8` | 킬 18번·"보수성 부족" 기준 |
| `risk.min_calib_peak_kw` | `1.0` | 이 미만이면 `not_assessed_low_load` |
| `ess.forecast_input` / `compare_inputs` / `compare_new_chargers` | `"p90"` / `true` / `3` | 8-B 입력과 AI 효과 비교 |
| `priority.axes` / `weights` / `min_axes` / `risk_metric` | `["risk","equity"]` / `{"risk":0.5,"equity":0.5}` / `2` / `"auto"` | 두 축 결합. 레거시는 `legacy_weights`·`legacy_min_axes`·`ahp_matrix` |
| `headline.risk_threshold` | `0.1` | 숫자 3(a) 기준(10일에 하루 이상 경보). 팀 확정 대상 |
| `scenario.*` | 위 8-F 기준값, `years [2028,2030]`, `k {저:0.5, 중:1.0}`, `growth_months 36`, `beta_bounds [0.3,2.0]`, `n_boot 999` | 8-F |
| `paths.weather` · `paths.holidays` · `paths.ev_history` · `paths.hdong_bjd` | — | 기온 CSV · 공휴일 달력 · 전기차 등록 이력 · 행정동→법정동 대응표. `ev_registration`이 비면 8-C도 이력의 기준월 값을 쓴다 |

### 킬 크라이테리아 — 설계 문서 번호 ↔ 코드 번호

| 설계 문서 | 코드 | 항목 | v11 판정 |
|---|---|---|---|
| 9 | 9 | ESS LP(scipy) | 그대로 |
| 10 | 8 | CAN 식별번호 | 그대로 |
| 11 | 2 | T_r 분포(활성화 후보) | `commerce=false`면 "해당 없음"(3·4·6·10번 SHC·MDE 항목도 같음) |
| 12 | 12 | 접근성 지역키 매칭 | 그대로(ev_history 경로 지원) |
| 13 | 13 | 계절 커버리지 | 그대로 |
| 14 | 14 | 가중치/AHP | v11: 가중치 유효성(합 1·음수 없음), AHP는 레거시에서만 |
| 15 | 15 | 강건 상위군 | 대응 문구 "축별 순위표 2장"(`s8e_rank_risk`·`s8e_rank_equity`) |
| — | 16 | 원천 수록 기간 | `commerce=false`면 "002를 001과 대조할 수 있는 기간" |
| 17 | 17 | ML 라이브러리 | 폴백 경로 표시 |
| 18 | 18 | AI P50 MAE ≤ 기준 모델, P90 적중률 ≥ 80% | 실패 시 "AI 개선 없음"(파이프라인 실행 후 판정) |
| 19 | 19 | 기온 예보·발표 시각 | 예보 없음 경고, 마감 뒤 발표가 대부분이면 실패 |
| 20 | 20 | 시나리오 매핑률·β·k_goal | 매핑 실패 10% 초과 경고, 30% 초과 실패 |
| 21 | 21 | 급증위험 0 비율 ≥ 50% | 경고 + 피크비율 대체 표시 |

판정 값은 통과·경고·실패·오류·**해당 없음**이다.

### 반출 규칙 추가분

새 표(`s1_channel_share`·`s8a_*`·`s8b_ai_effect*`·`s8e_rank_*`·`s8f_*`·`s9_headline`)도 같은 규칙이다: 좌표 열이 든 표는 저장 거부,
법정동 행은 대표 고객호수 < `output.min_cell_count`면 PNG에서 `—`, 법정동 열이 없는 합계·요약표(채널 비교·경보 요약·AI 효과 요약·
시나리오 요약·발표 숫자·예측 지표의 전체 행)는 소표본 법정동을 **빼고 다시 계산한 값**을 PNG에 싣는다(차감 역산 방지). 기온 지점
좌표(`station_lat`·`station_lon`)는 입력 전용이며 표에 싣지 않는다. 안심구역 안에서는 태블로로 CSV(`bjd_code` 키)를 경계와 결합해
지도를 만들 수 있으며, 반출하는 그림에도 이 규칙을 그대로 적용한다.

### 반입 묶음 (v11)

**5.csv·9.csv 받기:** `python tools/fetch_public_inputs.py 2024-01-01 2025-12-31 --out dist/import`(인터넷 되는 곳).
- `5.csv` = 전력거래소 EPSIS 시간별 SMP(육지). EPSIS "1시"(00~01시 구간)를 시간 시작 시각 00:00으로 바꿔 저장한다. 주말·휴일 한낮의 0원은 실제 가격(태양광 과잉)이다.
- `9.csv` = Open-Meteo(키 없음, CC BY 4.0), 서울 ASOS 108 지점 좌표 한 점. 예보는 **기상청(KMA) 예보모델이 대상 시각 48시간 전에 낸 값**이고 `fcst_issued_at = timestamp − 48h`로 둬 전날 18시 마감을 항상 지킨다. 예보 보관은 2025-02-28부터라 그 전 예보 칸은 비어 있다. 실측은 **ERA5 재분석(기상청 ASOS 아님)**이며 `observed` 참고 모델에만 쓴다. 설계서의 1순위 출처(기상청 ASOS·동네예보 과거자료)는 로그인·API 키가 필요해, 팀이 확보하면 같은 열로 바꿔 넣는다.

**데이터 파일명은 이전 반입(2026-09-21 V10: `dataansimbundle.py`·`bjdmaster.csv` 등)과 겹치지 않게 숫자로 매긴다.** 코드 모듈은 원래 이름(이전에는 번들 안에만 있었음).
`tools/check_import_bundle.py`가 이전 이름과 겹치면 위반으로 잡는다.

| 반입 이름 | 내용 | 설정 경로 |
|---|---|---|
| 코드 모듈 20개(`pipeline.py` 등) | 원래 이름 그대로 따로 반입 | — |
| `fieldtemplate.txt` | 한 폴더용 설정 원본(내용은 JSON). 센터에서 `field.txt`로 복사해 KEPCO·CAN 경로와 평가 시작일을 채움 | — |
| `holidays.txt` · `requirements.txt` · `README.txt` | 공휴일 달력(내용 JSON) · 패키지 목록 · 이 README | `paths.holidays` |
| `checksums.txt` | 위 파일들의 SHA-256. 일부만 옛 버전이거나 깨지면 `check_import_bundle.py`가 잡음 | — |
| `2.csv` | 법정동코드 마스터(`build_reference_files.py`의 bjdmaster) | `paths.bjd_master` |
| `3.csv` | 법정동 코드대응(bjdcrosswalk) | `paths.bjd_crosswalk` |
| `4.csv` | 법정동 중심점(bjdcentroids) | `paths.emd_centroids` |
| `5.csv` | SMP | `paths.smp` |
| `6.csv` | 공용 충전소 위치 | `paths.access_stations` |
| `7.csv` | 행정동별 전기차 등록 월별 이력(OA-21236) | `paths.ev_history` |
| `8.csv` | 행정동→법정동 대응표(가중치) | `paths.hdong_bjd` |
| `9.csv` | 기온 실측·과거 예보 | `paths.weather` |
| `10.ttf` | 한글 폰트(서버에 없을 때만) | `paths.font` |

**코드는 파일마다 따로 반입한다**(알아보기 쉽고 반입 심사에 유리). 반입 포털이 json·yaml·md를 받지 않아 설정·달력·README는 내용 그대로 `.txt`로 담는다(코드는 확장자가 아니라 내용으로 읽는다). 올린 파일은 센터에서 **한 폴더에 평평하게** 두고, 그 폴더에서 `run_checks("field.txt")`·`run("field.txt")`를 부른다. 파일 개수 제한은 없다("최대 10개"는 확인 결과 규정이 아님), 총 50MB. `python tools/make_import_bundle.py [폴더]`가 이 구성을 만들고 새 폴더에 복사해 해시·20개 모듈 import·설정 읽기를 확인한다.
SHC 파일은 기대하지 않는다. 8-C 전기차 대수는 `evhistory`+`hdongbjd`의 기준월 값을 써서 `evregistration.csv`를 따로 반입하지 않는다.
새 파일명은 반입 규칙(영문·숫자만)을 따른다. 데이터는 이전 반입 이름과 겹치지 않게 숫자(2~9.csv, 10.ttf)로 둔다.

**공휴일 달력:** `python tools/fetch_holidays.py 2024 2025`(인터넷 되는 곳, 서비스키 환경변수 `DATAGOKR_SERVICE_KEY`) →
`config/holidays.yaml`. PyYAML이 없을 수 있어 **JSON 문법**(= 유효한 YAML)으로 저장한다. 휴일을 규칙으로 만들지 않으며, 대체·임시공휴일·
선거일 누락은 관보로 보완해 `source`에 적는다. `config/holidays.yaml`에는 2024~2025 관공서 공휴일 38일(대체·임시공휴일·선거일 포함, Nager.Date API·정부 발표·언론 보도로 대조)과 유형이 들어 있다. 특일 정보 API 키가 생기면 `fetch_holidays.py`로 다시 대조한다.

### mock 실행 결과 (2026-09-21, seed 42) — 동작 확인용이며 성능 근거가 아니다

`mock_data.py --energy`는 SHC 없이 서울 3개 자치구 이름·코드(종로구·중구·용산구, 법정동 이름은 가상) 30개 법정동, 2025-05~12 일별
부하(기온·요일·휴일·추세 반응을 **심어 둠**), 기온(전날 17시·20시 발표 예보), 공휴일 달력, 행정동 단위 전기차 이력·대응표를 만든다.
`--with-shc`로만 SHC를 만든다.

| 확인 | 결과 |
|---|---|
| 전체 실행 | SHC 없이 전 단계 완료, 상권 단계 "비활성" 1행. 킬 21항목 중 오류 0 |
| 8-A(sklearn 폴백, 기온 forecast, 검증 56일) | MAE persistence 0.542 · 기준 0.486 · AI P50 0.433, 증가일 오차 기준 0.387 · AI 0.295, 피크 시간 오차 기준 1.487 · AI 1.496, P90 적중률 **0.77(80% 미달 → 킬 18 실패, "AI 개선 없음" 표시)**, 법정동 27/30곳에서 AI 우세 |
| 급증 경보(상한 1.2배, 평가 28일) | 급증위험 0인 동네 93% → 위험 축 **피크비율로 대체**(킬 21 경고). 1.1배에서 AI 정밀도 0.10·재현율 0.23, 기준 모델 경보 0건. 12-25(단일 공휴일)는 기준 모델 실패로 비교에서 30법정동·일 제외 |
| AI 효과(증설 3대) | 1.2배 평균 0.21(대상 10/15곳), 1.1배 0.32, 1.3배 0.70(대상 2곳). P90 운전의 과잉 방전이 기준 모델보다 큼(1.2배 81.8 대 9.3 kWh) |
| 8-E 순위 대상 | **30/30곳(100%)** — V10은 절반이 "부하 미평가". 급증위험·형평성 상관 피어슨 −0.15 |
| 8-F | 매핑 실패 0.35%(심어 둔 미매핑 행정동), β 구간이 0 포함 → β=1(0.8·1.2 민감도), k_goal 1.22. mock 증가율이 커서 2030 중 시나리오 30곳 모두 현재 1.2배 상한 초과 |

### 미구현(남은 것)

지표 3(자가충전 제약 주거 비율), 300/500m 커버리지(8-D), MCLP, DEM 경사 보정, 실제 GIS 지도(안심구역 태블로 경로 검토).
lightgbm 실경로는 설치 환경에서만 검증된다(테스트는 `importorskip`).

### 팀이 결정·확인할 것

1. 숫자 3 헤드라인 문구, `headline.risk_threshold`(0.1).
2. KEPCO_002 유지 여부 — `s1_channel_share`·킬 7·16으로 판단(mock은 002 ⊆ 001).
3. `scenario.national_base`·`seoul_base`를 통계누리 원자료로 대조(기준월 2026-08로 옮길지).
4. 반입 자료 범위: 경계 도형·공동주택 자료를 추가로 가져갈지(개수 제한은 없음, 총 50MB).
5. 설계 문서 3-2절의 "현재 기본 결합 시나리오는 공용 완속충전기 3대 증설" 문장은 v11.3(증설 0대)과 어긋나므로 문서 수정 필요.
6. 8-A 동네 특성: 설계서의 "교정기간 평균 부하"를 누설 방지를 위해 "학습기간 평균 부하"로 바꿨다.

## V10 기록(이력) — 아래는 V10 기준 서술이며 위 v11.3 절이 우선한다

## 실행

로컬(PowerShell, 저장소 루트):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe mock_data.py --out data/mock --energy
.venv\Scripts\python.exe killcriteria.py --config config/mock.json   # v11: mock 은 --energy 로 생성
.venv\Scripts\python.exe pipeline.py --config config/mock.json
.venv\Scripts\python.exe -m pytest -q tests
```

현장(안심구역 JupyterLab):

1. `config/fieldtemplate.json` → `config/field.json`, `config/industrycodestemplate.json` → `config/industry_codes.json` 복사.
2. `field.json` 의 `paths` 를 실제 경로로 바꾸고, 헤더가 다르면 `columns` 에 그 항목만 적는다(접두 일치 허용).
3. **첫날:** `from killcriteria import run_checks; run_checks("config/field.json")` — 업종 코드가 비어 있어도 돈다.
4. `TB_SHC_TOBU_CODE.csv` · `TB_SHC_CODE.csv` 로 `industry_codes.json` 을 채운 뒤 `from pipeline import run; run("config/field.json")`.

5. 8-A/8-B는 날짜가 보존된 1시간 원자료와 최소 12주 연속 이력이 필요하다. `energy.evaluation_start`는 현장에서 반드시 지정하고, 공휴일 달력·후보 장치 사양을 자료에 맞춰 확인한다. 로더는 검증·교정·평가에 필요한 기간만 읽는다. 월 집계만 있으면 예측은 실패로 남기며 실측으로 대체하지 않는다.
6. SMP는 `timestamp,smp`(KST 시간 시작, 원/kWh) CSV로 정규화한 뒤 `paths.smp`를 지정한다. 미입력은 피크 목적함수와 비용 결측이다. 과거 SMP는 사후 가격 평가이며 사전 이용 가능 시점은 별도 확인한다.
7. 현재 신청에서 제외한 KEP_007은 현장 템플릿의 `null`을 유지한다. 해당 선택 단계 생략은 의도한 부분완료이며, mock 전체 실행은 기존 경로 회귀 검증을 위해 합성 KEP_007을 포함한다.
8. **반입 묶음:** 반입 가능한 형식은 `.py`·`.csv`·`.txt`·폰트(`.ttf`)뿐이고 **zip·json·md는 올릴 수 없다**(파일명은 영문 대·소문자와 숫자만, `_`·`-`·공백·한글 불가). 그래서 코드 전체를 **`dataansimbundle.py` 한 파일**에 원문 그대로 담았다(`python tools/make_import_bundle.py`로 생성). 센터 JupyterLab에서 `%run dataansimbundle.py`를 실행하면 `dataansim/` 폴더에 모듈 18개·설정 템플릿·README·requirements가 풀리고 SHA-256으로 검증한다. 풀린 뒤 `from killcriteria import run_checks`, `from pipeline import run`으로 실행한다. 반입 예: 번들 `.py` 1 + 데이터 CSV 6 + 폰트 1 = **8개**(최대 10개·총 50MB). 권장 이름: `dataansimbundle.py`, `bjdmaster.csv`, `bjdcrosswalk.csv`, `bjdcentroids.csv`, `smp.csv`, `evstations.csv`, `evregistration.csv`, `koreanfont.ttf`. `python tools/check_import_bundle.py <폴더>`로 형식·개수·용량·파일명을 점검한다.
9. RAM이 8GB 이하이면 `params.shc.sido`(시도 코드 앞 2자리 목록)와 `chunksize`를 줄여 SHC를 시도별로 나눠 읽는다(`kepco.sido`와 같은 취지, 청크 단위 필터).

산출물: `out_dir/png/`(반출용) · `out_dir/csv/`(현장 작업용, 반출 대상 아님) · `out_dir/pipeline.log`.

## 참고자료 생성

법정동 마스터·코드대응·중심점 참고자료는 `data/ref/` 원본에서 `tools/build_reference_files.py`로 필요할 때 생성한다. 출력 파일명은 반입 규정(영문·숫자만)에 맞춰 `bjdmaster.csv`·`bjdcrosswalk.csv`·`bjdcentroids.csv`로 고정이며 세 파일을 각각 반입한다(zip은 코드만). 이전 `dist/` 반입 ZIP은 V9 코드와 일치하지 않아 보관하지 않는다.

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
| 0 지역키 정합 | `bjdmapping.py` | `s0_bjd_match_rate` · `s0_bjd_unmatched` | 중단 |
| 1 시계열 집계 | `kepcoloader.py` | `s1_kepco_monthly` · `s1_kepco_band_share` | 중단 |
| 2 변화점 | `kepcoloader.py` | `s2_activation` · `s2_activation_examples` | 중단 |
| 3 처치 정제 | `canloader.py` · `kep007loader.py` | `s3_can_*` · `s3_treated_excl_base` · `s3_kep007_stock` | **격리**(`s3_can_skipped`) |
| 4 Y_ddd | `identification.py` | `s4_ydd_panel` · `s4_skipped_*` · `s4_shc002_quality` | 중단 |
| 4.5 MDE | `identification.py` | `s45_mde` — 실제 처치 수 가정의 주·위약·할인 후 경험 SE와 MDE | 격리 |
| 5 이벤트 스터디 | `identification.py` | `s5_event_main` · `s5_pretrend_*` · `s5_event_regression` · `s5_s6_event_study` | 중단 |
| 6 위약 | `identification.py` | `s6_event_placebo` · `s6_discount_by_k` · `s6_post_summary` | 중단 |
| 6.5 처치오염 | `diagnostics.py` | `s65_contamination` | 격리 |
| 7 CATE | `heterogeneity.py` | `s7_features` · `s7_cate_region` · `s7_cate_validation` · `s7_subgroup_cells`/`s7_importance` | 격리 |
| 8 처방 | `loadaxis.py` | `s8_load_concentration` · `s8_quadrants` · `s8_quadrants_plot` | 격리 |
| 8-A 일별 예측 | `loadforecast.py` · `kepcoloader.py` | `s8a_validation` | 격리·실패 표시, 실측 대체 금지 |
| 8-B ESS | `essoptimizer.py` | `s8b_scenarios` · `s8b_schedules` · `s8b_comparison` · `s9_public_review` | 격리·행별 제약/예측/가격 상태 기록 |
| 8-C 접근성 | `equityaccess.py` | `s8c_accessibility`(2SFCA + 지표 1 `ev_per_charger` + 지표 2 `nearest_charger_km`) | 격리·외부자료 없으면 생략. 지표 1·2는 **법정동 중심점 근사**다: 충전소를 가장 가까운 중심점에 배정하며 폴리곤 공간조인·격자점이 아니다(행정경계 근처 충전소는 이웃 동네로 배정될 수 있음). **지표 3(자가충전 제약 주거 비율)은 미구현 — LH KLH_001이 신청 범위에 없어 신청 범위 결정 필요.** 300/500m 커버리지 비율은 8-D 후속(미구현) |
| 9-H 발표 3숫자 | `headline.py` | `s9_headline` | 격리·8-B 또는 8-C 결과 필요. 숫자 1 = 충전기 1기당 EV 수의 상위10%÷하위10% 배수, 숫자 2 = ESS 미적용 평균 상한 초과 kWh, 숫자 3 = 잔여 초과 kWh와 SMP 기준 비용 차이. 각 행에 "증설 N대·상한 배율·이용률 배율" 가정과 상한 배율 범위를 병기. PNG는 소표본 법정동 기여분 제외 |
| 8-E 결합점수 | `priorityscore.py` | `s8e_priority`(순위 대상만) · `s8e_not_ranked`(부하 미평가 — 접근성 부족만 확인) · `s8e_priority_summary`(풀 크기·안전/경제성 축 상관) | 격리·8-B/8-C 없으면 생략. 세 축이 모두 있는 동네만 순위·`robust_top`·킬 15번 분모에 든다(`params.priority.min_axes`, 기본 3). 결측 사유는 `not_assessed_low_concentration`(8-B 대상 아님)과 `data_missing`. 안전·경제성 상관 ≥ `params.priority.redundant_corr`(기본 0.9)이면 실행 요약에 "사실상 두 축" 경고 |
| 9 반출 | `outputs.py` · `pipeline.py` | `s0_run_summary` · `s9_manifest` | — |
| 킬 크라이테리아 | `killcriteria.py` | `kill_criteria/png/k_kill_criteria` | 항목별 `오류`. 16번 = 원천 수록 기간(활성화 창 끝과 비교) |

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
| 4사분면 문턱 | CATE·집중도 모두 중앙값 | `loadaxis.cate_cut`/`conc_cut` (`"zero"` 또는 숫자 가능) |
| 분석 지역 단위 | `params.analysis_level`: `"emd"`(법정동, 기본) · `"sigungu"`(시군구 폴백) | 한전 읍면동이 행정동이어서 법정동과 못 맞출 때(킬 5번 실패) `"sigungu"`로 다시 실행한다. 지역키는 시군구5자리+`00000`의 10자리 `bjd_code`이고, 한전은 시도+시군구로만 매칭하며 신한카드·EV는 코드를 시군구로 접는다. 읍면동 시간별 값은 시군구로 합산하고 소표본 억제용 고객호수도 합(공개 값을 만든 인원)으로 잡는다. **300/500/800m 2SFCA와 중심점 근사는 시군구 규모에서 의미가 없어 8-C·8-E는 생략**되고 8-B(안전·경제성)와 9-H 숫자 2·3은 나온다. 지역이 적으면(수십 곳 미만) MDE(4.5)·처치/대조 표본 부족으로 상권 단계가 생략될 수 있다. 해상도가 낮아진 결과이므로 "시군구 단위 잠정 검토"로만 보고한다. 행정동→법정동 배분표 방식(②)은 미구현 |
| 001/002 | 합산하지 않음. 활성화 시점은 `kepco.source`(제공 설정 002), 전력축은 001. **주의: `config.py` 기본값은 `kepco.source="001"`이고 `fieldtemplate.json`·`mock.json`은 `"002"`다(통일 여부는 팀 확인 필요).** 산출물 제목은 `activation.source_label`(기본: 002 "공용(사업자 채널) 충전 활성화", 001 "충전 활성화(전체)"). KEPCO_002는 가공일자(20250825)가 제공기간 끝(20251231)보다 이르므로 킬 크라이테리아 16번이 원천별 마지막 수록 달을 창 끝과 비교해 사후 `ratio_months`를 못 채우는 달 수를 적고, `activation.clip_to_data=true`면 창 끝을 마지막 수록 달로 줄인다 | 실제 포괄 범위·사업자 채널 대응은 현장 명세 확인 |

## 공학 산출물 해석

- 상한은 평가 이전 28일 기존 충전 부하 최대값의 1.1/1.2/1.3배다. 변압기 용량 추정값이 아니다. 증설은 0/2/3/5대, 이용률은 CAN 상대 모양 또는 설정 모양에 가정 배율을 적용한다.
- ESS는 이산 후보에서 초기·종단 에너지, SOC 사용 범위, 충방전 효율·출력, 재충전, 순부하 0 이상을 검증해 선택한다. 전역 최소 비용 용량이 아니다. 하루씩 독립된 실험이며 열화·설비비를 제외한다.
- forecast와 oracle에 같은 사전 선정 용량을 사용한다. 예측 오차로 실측 적용 시 상한·종단잔량을 못 지키면 그대로 보고한다. SMP 비용 차이는 실제 전기요금·교체비 절감이 아니다.
- **감사 반영(리플맵4.2 잔여 지적, 9/19):** ① 사전추세로 지역을 제외하는 옵션(`pretrend_action="exclude"`)을 삭제했다 — 표시(유보)만 하며 지역별 검정은 검정력이 낮다. ② CATE는 BLP 교정 검정(기울기>0, p≤`heterogeneity.blp_alpha` 0.10)을 통과해야 보고한다. 미충족이면 `s7_cate_region`의 `cate`가 비고 서브그룹·중요도 표를 내지 않으며, 4사분면은 부하 집중도만으로 나눈다(`⑤/⑥ (CATE 미보고)`). ③ 6단계 위약은 "위약이 0과 구분되는가"(p<0.05)를 판정 기준으로 삼아 주 결과 해석을 유보하고, 할인 후 값은 참고로 부트스트랩 CI와 함께 낸다. ④ 추정량 이름은 "충전 활동 활성화 시점의 외지 대기소비 매출 변화"이며 "충전소 설치 효과"라고 부르지 않는다. **미반영(후속):** 계절 조정·순열 추론(T_r 오차), BJS imputation 폴백, 인접 지역 도넛·시군구 클러스터·wild bootstrap, 심은 효과 복원 테스트.

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
