"""킬 크라이테리아 자동 체크 — 첫 방문 반나절(목표 30분) 안에 한 번에 판정.

v11: 상권 단계(stages.commerce=false, 기본)에서는 SHC·T_r 항목(2·3·4·6·10)을 "해당 없음"으로 표시하고 SHC 경로가
비어 있어도 실패하지 않는다. 16번은 "KEPCO_002를 001과 대조할 수 있는 기간" 점검이 된다.

실행:
    .venv\\Scripts\\python.exe killcriteria.py --config config/mock.json
    (JupyterLab) from killcriteria import run_checks; run_checks("config/field.json")

업종 코드 매핑(industry_codes)이 아직 비어 있어도 돈다 — 첫 방문에 코드값을 확인하는 것도 이 스크립트의 목적이다.
대용량 원천은 표본만 읽는다(params.kill.sample_rows · compare_rows · kepco_max_chunks).
판정: 통과 / 경고 / 실패 / 오류(파일·컬럼 문제) / 해당 없음. 결과는 out_dir/kill_criteria/{csv,png}/k_kill_criteria.*
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

import bjdmapping
import canloader
import equityaccess
import identification
import kepcoloader
import loadforecast
import loadscenario
import priorityscore
import weatherloader
from common import keep_region, log, mi_to_ym, setup_logging, ym_to_mi
from config import deep_merge, load_config, resolve_region, select_kepco_source
from outputs import OutputWriter, setup_korean_font

PACKAGES = [
    ("pandas", True, ""), ("numpy", True, ""), ("sklearn", True, "DBSCAN·BallTree·RandomForest"),
    ("matplotlib", True, "PNG 반출"),
    ("ruptures", False, "없으면 내장 PELT(같은 L2 목적함수 정확해)"),
    ("econml", False, "없으면 2x2 서브그룹 폴백"),
    ("linearmodels", False, "없어도 됨 — 내장 CS·FE 회귀"),
    ("statsmodels", False, "없어도 됨 — 내장 클러스터 SE·p값"),
    ("lightgbm", False, "없으면 sklearn HistGradientBoosting 분위수 → 그것도 없으면 기준 모델(킬 17)"),
]


def run_checks(config_path, params_override=None, paths_override=None, write=True):
    cfg = load_config(config_path, require_industry=False)
    if params_override:
        cfg["params"] = resolve_region(deep_merge(cfg["params"], params_override))
    bjdmapping.set_analysis_level(cfg["params"]["analysis_level"])
    if paths_override:
        cfg["paths"].update({k: (str(Path(v).resolve()) if v else None) for k, v in paths_override.items()})
    P, C, paths, ind = cfg["params"], cfg["columns"], cfg["paths"], cfg["industry"]
    K = P["kill"]
    commerce = bool(P["stages"]["commerce"])
    prefix = P["region"]["sido_prefix"]
    kepco_src, kepco_preliminary = select_kepco_source(paths, P)
    kepco_cols = C["kepco"] if kepco_src == "001" else C["kepco_cpo"]
    out_dir = Path(paths["out_dir"]) / "kill_criteria"
    setup_logging(out_dir)
    rows = []

    def add(no, item, verdict, evidence, action=""):
        rows.append({"번호": no, "항목": item, "판정": verdict, "근거": evidence, "조치": action})
        log.info("[%d] %s → %s | %s", no, item, verdict, evidence)

    def not_applicable(no, item, why="상권 단계 비활성(v11) — SHC·T_r 미사용"):
        add(no, item, "해당 없음", why)

    def guarded(no, item, fn):
        try:
            fn()
        except Exception as exc:
            add(no, item, "오류", f"{type(exc).__name__}: {str(exc)[:200]}", "경로·컬럼명(config columns) 확인")

    # 1. 패키지
    def check1():
        missing_req, missing_opt, present = [], [], []
        for name, required, note in PACKAGES:
            try:
                mod = importlib.import_module(name)
                present.append(f"{name} {getattr(mod, '__version__', '')}".strip())
            except Exception:
                (missing_req if required else missing_opt).append(f"{name}({note})" if note else name)
        if missing_req:
            add(1, "패키지 확인", "실패", f"필수 없음: {missing_req}", "필수 패키지 반입 신청")
        elif missing_opt:
            add(1, "패키지 확인", "경고", f"선택 없음: {', '.join(missing_opt)} / 있음: {', '.join(present)}",
                "폴백 경로로 진행")
        else:
            add(1, "패키지 확인", "통과", ", ".join(present))
    guarded(1, "패키지 확인", check1)

    # 2·5. KEPCO 활성화 후보 + 매핑 실패율 (한 번 로드해 둘 다 판정)
    kepco_state = {}

    def load_kepco():
        master = keep_region(bjdmapping.load_bjd_master(paths["bjd_master"], C["bjd"]), prefix)
        raw = kepcoloader.load_kepco_monthly_hour(paths[f"kepco_{kepco_src}"], kepco_cols, P["kepco"], f"KEPCO_{kepco_src}",
                                                   max_chunks=K["kepco_max_chunks"])
        mh, rate, unmatched, fail = kepcoloader.attach_bjd(raw, master, P["bjd"]["fail_warn_rate"])
        kepco_state.update(master=master, mh=mh, fail=fail, n_regions=len(raw[["sido", "sigungu", "emd"]].drop_duplicates()),
                           unmatched=unmatched)

    def check2():
        load_kepco()
        a = P["activation"]
        daily = kepcoloader.daily_series(kepco_state["mh"])
        cands = kepcoloader.simple_candidates(daily, a["window"], a["min_ratio"], a["ratio_months"])
        n = len(cands)
        ev = f"{'~'.join(a['window'])} 전후 {a['ratio_months']}개월 평균비 >= {a['min_ratio']} 고유 법정동 {n}곳 (set 집계)"
        if n < K["es_min_regions"]:
            add(2, "활성화 후보 지역 수", "실패", ev, f"{K['es_min_regions']}곳 미만 — 이벤트 스터디 포기")
        elif n < K["cf_min_regions"]:
            add(2, "활성화 후보 지역 수", "경고", ev, f"{K['cf_min_regions']}곳 미만 — Causal Forest 포기, 2x2 서브그룹")
        else:
            add(2, "활성화 후보 지역 수", "통과", ev)
    if commerce:
        guarded(2, "활성화 후보 지역 수", check2)
    else:
        not_applicable(2, "활성화 후보 지역 수(T_r 분포)")

    # 3·4·6. SHC002 표본
    shc_state = {}

    def load_shc():
        scan = identification.scan_shc002(paths["shc002"], C, ind, P["shc"], nrows=K["sample_rows"], extra_stats=True)
        shc_state.update(scan=scan, stats=scan["stats"])

    def check3():
        load_shc()
        vals = shc_state["stats"]["tizo_values"]
        ev = f"TIZO_CLCD {len(vals)}개: {vals} (앞 {K['sample_rows']:,}행)"
        if len(vals) < K["min_tizo"]:
            add(3, "시간대구간 수", "경고", ev, f"{K['min_tizo']}개 미만 — 시간대 축 포기")
        else:
            add(3, "시간대구간 수", "통과", ev)
    if commerce:
        guarded(3, "시간대구간 수", check3)
    else:
        not_applicable(3, "시간대구간 수")

    def check4():
        if "stats" not in shc_state:
            raise RuntimeError("SHC002 표본 로드 실패(3번 참조)")
        vals = shc_state["stats"]["dist_values"]
        ev = f"IFW_DISTC_SECT_CD {len(vals)}개: {vals}"
        resident = {str(c) for c in ind["resident_dist_codes"]}
        if len(vals) <= 1:
            add(4, "유입거리구간 정의", "실패", ev, "값 1개 — 층1(외지/거주자 대조) 불가")
        elif resident and not resident & set(vals):
            add(4, "유입거리구간 정의", "경고", ev + f" / 설정 거주자 코드 {sorted(resident)} 가 데이터에 없음",
                "industry_codes.resident_dist_codes 수정")
        elif not resident:
            add(4, "유입거리구간 정의", "경고", ev, "거주자 구간 미설정 — 코드 정의서 확인 후 resident_dist_codes 입력")
        else:
            add(4, "유입거리구간 정의", "통과", ev + f" / 거주자 = {sorted(resident)}")
    if commerce:
        guarded(4, "유입거리구간 정의", check4)
    else:
        not_applicable(4, "유입거리구간 정의")

    def check5():
        if "fail" not in kepco_state:
            load_kepco()
        fail = kepco_state["fail"]
        n_unmatched = len(kepco_state["unmatched"])
        ev = (f"KEPCO_{kepco_src}{' 잠정' if kepco_preliminary else ''} 지역 "
              f"{kepco_state['n_regions']}곳 중 미매칭 {n_unmatched}곳({fail:.1%}) · 원천 지역명은 반출표에서 제외")
        if fail >= P["bjd"]["fail_warn_rate"]:
            add(5, "한전 읍면동 매핑 실패율", "실패", ev,
                "행정동 기준일 가능성 — params.analysis_level=\"sigungu\"(시군구 폴백)로 다시 실행하거나 행정동↔법정동 매핑표 확보")
        elif fail > 0.10:
            add(5, "한전 읍면동 매핑 실패율", "경고", ev, "미매칭 목록 사람 검토")
        else:
            add(5, "한전 읍면동 매핑 실패율", "통과", ev)
    guarded(5, "한전 읍면동 매핑 실패율", check5)

    def check6():
        if "stats" not in shc_state:
            raise RuntimeError("SHC002 표본 로드 실패(3번 참조)")
        rate = shc_state["stats"]["mask_rate"]
        ev = f"SALE_AMT 결측/마스킹 {rate:.1%} (앞 {shc_state['stats']['rows']:,}행)"
        if rate > K["mask_max_rate"]:
            add(6, "마스킹 비율", "경고", ev, "PK 축소(성별·연령·소득 제거) 재신청 권고 · use_log1p 검토")
        else:
            add(6, "마스킹 비율", "통과", ev)
    if commerce:
        guarded(6, "마스킹 비율", check6)
    else:
        not_applicable(6, "마스킹 비율")

    # 7. KEPCO_001/002 포함관계
    def check7():
        if not paths["kepco_001"] or not paths["kepco_002"]:
            not_applicable(7, "KEPCO_001/002 포함관계", "001·002 중 하나가 없어 포함관계 대조 불가")
            return
        kp = dict(P["kepco"], chunksize=K["compare_rows"])
        keys = ["sido", "sigungu", "emd", "mi", "hour"]
        a = kepcoloader.load_kepco_monthly_hour(paths["kepco_001"], C["kepco"], kp, "KEPCO_001", max_chunks=1)
        b = kepcoloader.load_kepco_monthly_hour(paths["kepco_002"], C["kepco_cpo"], kp, "KEPCO_002", max_chunks=1)
        m = b.merge(a, on=keys, how="left", suffixes=("_002", "_001"), indicator=True)
        matched = m["_merge"] == "both"
        share = matched.mean() if len(m) else float("nan")
        both = m[matched]
        le_kwh = (both["kwh_002"] <= both["kwh_001"] * 1.0001).mean() if len(both) else float("nan")
        le_cust = (both["cust_sum_002"] <= both["cust_sum_001"]).mean() if len(both) else float("nan")
        ev = f"002 셀 {len(m):,} 중 001 에 존재 {share:.1%} · kWh 002<=001 {le_kwh:.1%} · 고객호수 002<=001 {le_cust:.1%}"
        if share >= 0.9 and le_kwh >= 0.99 and le_cust >= 0.99:
            add(7, "KEPCO_001/002 포함관계", "통과", ev + " → 002 ⊆ 001 정황", "그래도 합산 금지(001 이 이미 포함)")
        else:
            add(7, "KEPCO_001/002 포함관계", "경고", ev + " → 불명확", "합산 금지 · 001 단독 사용")
    guarded(7, "KEPCO_001/002 포함관계", check7)

    # 8. CAN 식별번호
    def check8():
        can = canloader.load_can_m(paths["can_m"], C, P["can"], nrows=K["sample_rows"])
        verdict, evidence = canloader.verify_join_key(can, P["can"])
        ev = " · ".join(f"{r['항목']}={r['값']}" for _, r in evidence.iterrows() if r["항목"] != "판정")
        if verdict == "individual":
            add(8, "CAN verify_join_key", "통과", f"individual | {ev}", "경로 A(개별 차량 클러스터링)")
        else:
            add(8, "CAN verify_join_key", "경고", f"{verdict} | {ev}", "경로 B(법정동 밀집도, 보조 근거)")
    guarded(8, "CAN verify_join_key", check8)

    # 10. 실제 처치 수 기준 검출 가능 최소효과
    def check10():
        if "scan" not in shc_state:
            load_shc()
        # R4(2026-09-19): 본 분석(pipeline)은 P["kepco"]["source"]의 원천을 쓰는데, 이 항목은
        # KEPCO_001을 하드코딩해 원천이 다르면(현장 002) 처치 수가 실제와 어긋났다.
        src, cols = kepco_src, kepco_cols
        master = kepco_state["master"] if "master" in kepco_state else bjdmapping.load_bjd_master(paths["bjd_master"], C["bjd"])
        raw = kepcoloader.load_kepco_monthly_hour(paths[f"kepco_{src}"], cols, P["kepco"], f"KEPCO_{src}",
                                                   max_chunks=K["kepco_max_chunks"])
        mh, *_ = kepcoloader.attach_bjd(raw, master, P["bjd"]["fail_warn_rate"])
        activation = kepcoloader.detect_activation(kepcoloader.daily_series(mh), P["activation"])
        panel, _ = identification.build_ydd(shc_state["scan"]["cells"], ind, P["identification"]["use_log1p"])
        mp = P["mde"]
        note = "표본 기반 참고값 — 본 판정은 pipeline 4.5"
        try:
            result = identification.estimate_mde(
                panel, activation, P["identification"], mp["repetitions"], mp["seed"],
                P["activation"]["window"], mp["bias_se_multiple"])
        except ValueError as exc:
            # kepco_max_chunks·sample_rows 표본 제약으로 처치·대조 수가 부족할 수 있다 — 오류가 아니라 경고.
            add(10, "검출 가능 최소효과(MDE)", "경고", f"{note} · {exc}", "표본 확대 또는 pipeline 4.5 결과로 판단")
            return
        row = result.loc[result["구분"] == "할인 후"].iloc[0]
        verdict = "경고" if row["MDE"] > K["mde_warn"] else "통과"
        evidence = " · ".join(f"{r['구분']} {r['MDE']:.4f}" for _, r in result.iterrows())
        add(10, "검출 가능 최소효과(MDE)", verdict,
            f"{note} · MDE: {evidence} · 반복 {int(row['반복수'])}회 · 처치 {int(row['처치수_가정'])}곳",
            "상권 효과는 MDE 이상 여부만 보고" if verdict == "경고" else "")
    if commerce:
        guarded(10, "검출 가능 최소효과(MDE)", check10)
    else:
        not_applicable(10, "검출 가능 최소효과(MDE)")

    # 11. PNG 반출용 한글 폰트
    def check11():
        try:
            name = setup_korean_font(paths["font"], P["outputs"]["font_fallback"])
        except (OSError, ValueError) as exc:
            add(11, "한글 폰트", "실패", str(exc), "PNG 반출 불가 — 폰트 파일 반입 또는 서버 폰트 확인")
            return
        if name:
            add(11, "한글 폰트", "통과", f"사용 폰트: {name}")
        elif P["outputs"]["font_fallback"] == "ascii":
            add(11, "한글 폰트", "경고", "사용 가능한 한글 폰트 없음 — 영문 대체 라벨로 진행",
                "PNG 한글이 영문 라벨로 나옴. 한글이 필요하면 폰트 파일 반입")
        else:
            add(11, "한글 폰트", "실패", "사용 가능한 한글 폰트 없음",
                "PNG 반출 불가 — 폰트 파일 반입 또는 서버 폰트 확인")
    guarded(11, "한글 폰트", check11)

    if P["energy"]["enabled"]:
        def check9():
            try:
                import scipy
                from scipy.optimize import linprog
            except ImportError as exc:
                add(9, "ESS LP 실행 환경", "경고", str(exc), "그리디 전환 후 물리 제약 재검증")
                return
            result = linprog([1.0], bounds=[(0, 1)], method="highs")
            add(9, "ESS LP 실행 환경", "통과" if result.success else "경고",
                f"scipy {scipy.__version__} · HiGHS: {result.message}",
                "데이터별 실행가능성은 8-B에서 별도 검증")
        guarded(9, "ESS LP 실행 환경", check9)
    else:
        not_applicable(9, "ESS LP 실행 환경", "energy.enabled=false")

    # 16. 선택한 원천의 마지막 수록 달이 활성화 창 끝보다 이른가(KEPCO_002 는 가공일자가 제공기간보다 이르다)
    def check16_channel():
        """v11: 002(사업자 채널)를 001과 대조할 수 있는 기간 — 두 원천의 관측 일자가 겹치는 기간."""
        item = "원천 수록 기간(002↔001 대조 가능 기간)"
        if not paths["kepco_002"]:
            not_applicable(16, item, "paths.kepco_002 없음 — 002 미사용")
            return
        spans = {}
        for src, cols in (("001", C["kepco"]), ("002", C["kepco_cpo"])):
            dates = pd.to_datetime(kepcoloader.scan_observed_dates(paths[f"kepco_{src}"], cols, P["kepco"], f"KEPCO_{src}")["date"])
            if dates.empty:
                add(16, item, "경고", f"KEPCO_{src} 관측 일자를 못 읽음", "period 열·컬럼 설정 확인")
                return
            spans[src] = (dates.min(), dates.max())
        start, end = max(s[0] for s in spans.values()), min(s[1] for s in spans.values())
        ev = " · ".join(f"{k} {a:%Y-%m-%d}~{b:%Y-%m-%d}" for k, (a, b) in spans.items())
        if end < start:
            add(16, item, "경고", ev + " · 겹치는 기간 없음", "002 채널 비교(s1_channel_share) 불가 — 002 유지 여부 팀 결정")
        else:
            add(16, item, "통과", ev + f" · 대조 가능 {start:%Y-%m-%d}~{end:%Y-%m-%d}")

    def check16():
        src, cols = kepco_src, kepco_cols
        dates = kepcoloader.scan_observed_dates(paths[f"kepco_{src}"], cols, P["kepco"], f"KEPCO_{src}")
        if dates.empty:
            add(16, "원천 수록 기간", "경고", f"KEPCO_{src} 관측 일자를 못 읽음", "period 열·컬럼 설정 확인")
            return
        last = ym_to_mi(dates["date"].max())
        a = P["activation"]
        end = ym_to_mi(a["window"][1])
        label = f"KEPCO_{src} 마지막 수록 {mi_to_ym(last)} · 활성화 창 끝 {a['window'][1]}"
        if last < end:
            short = kepcoloader.window_shortfall(last, a["window"], a["ratio_months"])
            add(16, "원천 수록 기간", "경고", f"{label} · 사후 {a['ratio_months']}개월을 못 채우는 후보 달 {short}개",
                "activation.window 끝을 줄이거나 activation.clip_to_data=true")
        else:
            add(16, "원천 수록 기간", "통과", label)
    if commerce or not (paths["kepco_001"] and paths["kepco_002"]):
        guarded(16, "원천 수록 기간", check16)
    else:
        guarded(16, "원천 수록 기간(002↔001 대조 가능 기간)", check16_channel)

    csv_dir = Path(paths["out_dir"]) / "csv"

    def read_output(name):
        path = csv_dir / f"{name}.csv"
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"bjd_code": str}) if path.exists() else None

    if P["energy"]["enabled"]:
        # 17. ML 라이브러리 → 폴백 경로
        def check17():
            name, _ = loadforecast.select_model(P["forecast"]["model"])
            if name == "baseline":
                add(17, "AI 예측 라이브러리", "경고", "lightgbm·scikit-learn(분위수 손실) 모두 없음",
                    "8-A 는 기준 모델만 — 'AI 예측 미실행' 표시, 숫자 2 미실행")
            else:
                add(17, "AI 예측 라이브러리", "통과", f"사용 모델: {name}" + (" (lightgbm 없음 → sklearn 폴백)"
                                                                         if name == "sklearn" else ""))
        guarded(17, "AI 예측 라이브러리", check17)

        # 18. AI 성능 — P50 MAE ≤ 기준 모델, P90 적중률 ≥ 80%
        def check18():
            metrics = read_output("s8a_forecast_metrics")
            if metrics is None:
                add(18, "AI 성능(기준 모델 대비)", "경고", "8-A 산출물 없음", "파이프라인 실행 후 재확인")
                return
            ok, why = loadforecast.ai_improvement(metrics, P["forecast"]["min_p90_coverage"])
            add(18, "AI 성능(기준 모델 대비)", "통과" if ok else "실패", why,
                "" if ok else "'AI 개선 없음' 표시 · 숫자 2는 경보 비교만 보고 · 보수성 부족이면 경보·ESS에 표시")
        guarded(18, "AI 성능(기준 모델 대비)", check18)

        # 19. 기온 예보 확보·발표 시각
        def check19():
            item = "기온 예보·발표 시각"
            cutoff = P["weather"]["fcst_cutoff"]
            if not paths["weather"]:
                add(19, item, "경고", "paths.weather 없음", "기온 없는 모델(none)로 발표 숫자 — 실측 기온 모델은 참고만")
                return
            weather = weatherloader.load_weather(paths["weather"])
            n_fcst = int(weather["temp_fcst_c"].notna().sum())
            if n_fcst == 0:
                add(19, item, "경고", "예보 기온 열이 비어 있음(실측만)", "none·observed 두 모델 병기, 발표는 none 기준")
                return
            usable = weatherloader.forecast_temps(weather, cutoff)
            weatherloader.assert_forecast_before_cutoff(usable, cutoff)
            targets = weather.loc[weather["temp_fcst_c"].notna(), ["station_or_grid", "timestamp"]].drop_duplicates()
            share = len(usable) / max(1, len(targets))
            ev = f"예보 대상 시각 {len(targets):,} 중 전날 {cutoff} 이전 발표분 있음 {share:.1%} · 마감 뒤 발표 행은 제외"
            if share < 0.5:
                add(19, item, "실패", ev, "발표 시각이 마감 이후인 예보가 대부분 — 예보 모드 불가, none 으로 실행")
            else:
                add(19, item, "통과", ev)
        guarded(19, "기온 예보·발표 시각", check19)
    else:
        not_applicable(17, "AI 예측 라이브러리", "energy.enabled=false")
        not_applicable(18, "AI 성능(기준 모델 대비)", "energy.enabled=false")
        not_applicable(19, "기온 예보·발표 시각", "energy.enabled=false")

    if paths["ev_history"]:
        # 20. 시나리오 — 행정동→법정동 매핑률, β 안정성, k_goal 보정 가능 여부
        def check20():
            item = "시나리오 매핑·β·k_goal"
            if not paths["hdong_bjd"]:
                add(20, item, "실패", "paths.hdong_bjd 없음", "행정동→법정동 대응표(가중치) 확보")
                return
            sp = P["scenario"]
            hist = loadscenario.load_ev_history(paths["ev_history"], C["ev_history"], sp["fuel_value"], prefix)
            mapping = loadscenario.load_hdong_bjd(paths["hdong_bjd"], C["hdong_bjd"])
            _, fail = loadscenario.map_to_bjd(hist, mapping, base_month=sp["base_month"])
            parts = [f"매핑 실패 {fail:.1%}(기준월 전기차 기준)"]
            verdict = "실패" if fail > P["bjd"]["fail_warn_rate"] else "경고" if fail > 0.10 else "통과"
            beta = read_output("s8f_beta")
            if beta is None:
                parts.append("β·k_goal 은 파이프라인 실행 후 확인")
                verdict = "경고" if verdict == "통과" else verdict
            else:
                b = beta.iloc[0]
                parts.append(f"β {b['beta']:.2f}" + (f"(불안정: {b['reason']} → 1, 0.8·1.2 민감도)"
                                                    if str(b["fallback"]).lower() == "true" else ""))
                parts.append("k_goal 없음 — 고 시나리오 생략" if pd.isna(b["k_goal"]) else f"k_goal {b['k_goal']:.3f}")
                if pd.isna(b["k_goal"]) and verdict == "통과":
                    verdict = "경고"
            add(20, item, verdict, " · ".join(parts),
                "대응표 보완(미매핑 행정동 목록 확인)" if fail > 0.10 else "")
        guarded(20, "시나리오 매핑·β·k_goal", check20)
    else:
        not_applicable(20, "시나리오 매핑·β·k_goal", "paths.ev_history 없음")

    if P["priority"]["enabled"]:
        # 12. 외부 접근성 자료의 법정동 매칭
        def ev_from_history():
            sp = P["scenario"]
            hist = loadscenario.load_ev_history(paths["ev_history"], C["ev_history"], sp["fuel_value"], prefix)
            ev, _ = loadscenario.map_to_bjd(hist, loadscenario.load_hdong_bjd(paths["hdong_bjd"], C["hdong_bjd"]),
                                            base_month=sp["base_month"])
            return ev.loc[ev["ym"] == sp["base_month"], ["bjd_code", "ev_count"]]

        def check12():
            if bjdmapping.ANALYSIS_LEVEL == "sigungu":
                add(12, "접근성 지역키 매칭", "경고", "analysis_level=sigungu — 2SFCA(8-C)는 시군구 규모에서 의미가 없어 생략",
                    "형평성 축은 읍면동(법정동) 모드에서만 계산")
                return
            from_history = not paths["ev_registration"] and paths["ev_history"] and paths["hdong_bjd"]
            if not paths["access_stations"] or not (paths["ev_registration"] or from_history) or not paths["emd_centroids"]:
                add(12, "접근성 지역키 매칭", "경고", "접근성 입력 경로 미설정", "공개자료 반입 후 다시 실행")
                return
            cent = keep_region(bjdmapping.load_emd_centroids(paths["emd_centroids"], C["centroid"]), prefix)
            _stations, points = equityaccess.load_access_inputs(
                paths["access_stations"], paths["ev_registration"], C, cent, ev_counts=ev_from_history() if from_history else None)
            rate = float(points["ev_count"].notna().mean())
            verdict = "통과" if rate >= K["access_match_min"] else "실패"
            add(12, "접근성 지역키 매칭", verdict, f"중심점 기준 EV 등록자료 매칭 {rate:.1%}",
                "행정동→법정동 대응표 보완" if verdict == "실패" else "")
        guarded(12, "접근성 지역키 매칭", check12)

        # 13. 4계절 시계열 포함 여부
        def check13():
            if not paths["kepco_hourly"]:
                add(13, "계절 커버리지", "경고", "paths.kepco_hourly 없음", "계절 검증 보류")
                return
            if "master" not in kepco_state:
                load_kepco()
            hourly = kepcoloader.load_kepco_hourly(paths["kepco_hourly"], C["kepco"], P["kepco"],
                                                    kepco_state["master"])
            coverage = priorityscore.seasonal_coverage(hourly)
            verdict = "통과" if coverage["season_status"] == "검증 가능" else "경고"
            add(13, "계절 커버리지", verdict,
                f"{coverage['season_count']}/4계절 · 누락 {coverage['missing_seasons'] or '없음'}",
                "단일 평가기간 결과에 계절 미검증 표시" if verdict == "경고" else "")
        guarded(13, "계절 커버리지", check13)

        # 14. 설정 가중치 유효성(v11 두 축: 합 1·음수 없음). AHP 는 레거시 세 축에서만.
        legacy = list(P["priority"]["axes"]) == priorityscore.LEGACY_AXES

        def check14():
            if not legacy:
                w = priorityscore.validate_weights_v11(P["priority"]["weights"])
                add(14, "가중치 유효성", "통과", f"급증위험·형평성 {w.round(4).tolist()} (합 1, 음수 없음) · AHP 미사용(두 축)")
                return
            w = priorityscore.validate_weights(P["priority"]["legacy_weights"])
            matrix = P["priority"].get("ahp_matrix")
            if matrix is None:
                add(14, "가중치/AHP 일관성", "통과", f"설정 가중치 {w.round(4).tolist()} · AHP 입력 없음")
                return
            cr = priorityscore.ahp_consistency_ratio(matrix)
            add(14, "가중치/AHP 일관성", "통과" if cr < 0.1 else "실패", f"CR={cr:.4f}",
                "CR<0.1이 되도록 쌍대비교 재검토" if cr >= 0.1 else "")
        guarded(14, "가중치 유효성" if not legacy else "가중치/AHP 일관성", check14)

        # 15. 파이프라인 산출 후 강건 상위군 존재 확인
        def check15():
            path = Path(paths["out_dir"]) / "csv" / "s8e_priority.csv"
            if not path.exists():
                add(15, "민감도 강건 상위군", "경고", "8-E 산출물 없음", "파이프라인 실행 후 재확인")
                return
            result = pd.read_csv(path, encoding="utf-8-sig")
            pool = result[result["rank_eligible"].astype(str).str.lower().isin(["true", "1"])] \
                if "rank_eligible" in result else result
            if pool.empty:
                add(15, "민감도 강건 상위군", "경고", f"순위 대상 0곳(전체 {len(result)}곳)", "min_axes·8-B 평가 대상 확인")
                return
            share = float(pool["robust_top"].astype(str).str.lower().isin(["true", "1"]).mean())
            verdict = "통과" if share >= K["robust_top_min_share"] else "경고"
            add(15, "민감도 강건 상위군", verdict, f"순위 대상 {len(pool)}곳 중 {share:.1%}",
                ("축별 순위표 2장(s8e_rank_risk·s8e_rank_equity)을 결합순위 대신 병기" if not legacy
                 else "단일 결합순위 대신 축별 순위 병기") if verdict == "경고" else "")
        guarded(15, "민감도 강건 상위군", check15)

        # 21. 급증위험 0 동네가 순위 대상의 절반 이상인가(v11.3)
        def check21():
            item = "급증위험 변별력(0 비율)"
            if legacy:
                not_applicable(21, item, "레거시 세 축 모드")
                return
            result = read_output("s8e_priority")
            if result is None or "surge_risk" not in result:
                add(21, item, "경고", "8-E 산출물 없음", "파이프라인 실행 후 재확인")
                return
            share = float((pd.to_numeric(result["surge_risk"], errors="coerce") == 0).mean())
            metric = result["risk_metric"].iloc[0] if "risk_metric" in result and len(result) else "?"
            ev = f"순위 대상 {len(result)}곳 중 급증위험 0 {share:.1%} · 사용 위험 지표 {metric}"
            if share >= 0.5:
                add(21, item, "경고", ev, "위험 축을 피크비율(peak_ratio)로 대체했음을 표시(risk_metric=auto)")
            else:
                add(21, item, "통과", ev)
        guarded(21, "급증위험 변별력(0 비율)", check21)
    else:
        not_applicable(12, "접근성 지역키 매칭", "priority.enabled=false")
        not_applicable(13, "계절 커버리지", "priority.enabled=false")
        not_applicable(14, "가중치 유효성", "priority.enabled=false")
        not_applicable(15, "민감도 강건 상위군", "priority.enabled=false")
        not_applicable(21, "급증위험 변별력(0 비율)", "priority.enabled=false")

    table = pd.DataFrame(rows).sort_values("번호").reset_index(drop=True)
    if write:
        configured_font = paths["font"] if paths["font"] and Path(paths["font"]).is_file() else None
        writer = OutputWriter(out_dir, **P["outputs"], font_path=configured_font)
        # 오류 메시지에는 원본 값(조회기간 예)·센터 경로가 섞일 수 있어 PNG 에는 예외 종류만 싣는다(전문은 내부 CSV).
        error = table["판정"] == "오류"
        png = table.assign(근거=table["근거"].where(~error, table["근거"].str.split(":").str[0] + " — 상세는 내부 CSV"))
        writer.table(table, "k_kill_criteria", f"킬 크라이테리아 {len(table)}항목 판정", png_df=png)
    return table


def main():
    parser = argparse.ArgumentParser(description="킬 크라이테리아 자동 판정")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    table = run_checks(args.config)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
