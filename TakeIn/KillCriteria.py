"""킬 크라이테리아 자동 체크 — 첫 방문 반나절(목표 30분) 안에 8개 항목을 한 번에 판정.

실행:
    .venv\\Scripts\\python.exe kill_criteria.py --config config/mock.json
    (JupyterLab) from kill_criteria import run_checks; run_checks("config/field.json")

업종 코드 매핑(industry_codes)이 아직 비어 있어도 돈다 — 첫 방문에 코드값을 확인하는 것도 이 스크립트의 목적이다.
대용량 원천은 표본만 읽는다(params.kill.sample_rows · compare_rows · kepco_max_chunks).
판정: 통과 / 경고 / 실패 / 오류(파일·컬럼 문제). 결과는 out_dir/kill_criteria/{csv,png}/k_kill_criteria.*
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

import bjd_mapping
import can_loader
import identification
import kepco_loader
from common import log, setup_logging
from config import deep_merge, load_config
from outputs import OutputWriter

PACKAGES = [
    ("pandas", True, ""), ("numpy", True, ""), ("sklearn", True, "DBSCAN·BallTree·RandomForest"),
    ("matplotlib", True, "PNG 반출"),
    ("ruptures", False, "없으면 내장 PELT(같은 L2 목적함수 정확해)"),
    ("econml", False, "없으면 2x2 서브그룹 폴백"),
    ("linearmodels", False, "없어도 됨 — 내장 CS·FE 회귀"),
    ("statsmodels", False, "없어도 됨 — 내장 클러스터 SE·p값"),
]


def run_checks(config_path, params_override=None, paths_override=None, write=True):
    cfg = load_config(config_path, require_industry=False)
    if params_override:
        cfg["params"] = deep_merge(cfg["params"], params_override)
    if paths_override:
        cfg["paths"].update({k: (str(Path(v).resolve()) if v else None) for k, v in paths_override.items()})
    P, C, paths, ind = cfg["params"], cfg["columns"], cfg["paths"], cfg["industry"]
    K = P["kill"]
    out_dir = Path(paths["out_dir"]) / "kill_criteria"
    setup_logging(out_dir)
    rows = []

    def add(no, item, verdict, evidence, action=""):
        rows.append({"번호": no, "항목": item, "판정": verdict, "근거": evidence, "조치": action})
        log.info("[%d] %s → %s | %s", no, item, verdict, evidence)

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
        master = bjd_mapping.load_bjd_master(paths["bjd_master"], C["bjd"])
        raw = kepco_loader.load_kepco_monthly_hour(paths["kepco_001"], C["kepco"], P["kepco"], "KEPCO_001",
                                                   max_chunks=K["kepco_max_chunks"])
        mh, rate, unmatched, fail = kepco_loader.attach_bjd(raw, master, P["bjd"]["fail_warn_rate"])
        kepco_state.update(mh=mh, fail=fail, n_regions=len(raw[["sido", "sigungu", "emd"]].drop_duplicates()),
                           unmatched=unmatched)

    def check2():
        load_kepco()
        a = P["activation"]
        daily = kepco_loader.daily_series(kepco_state["mh"])
        cands = kepco_loader.simple_candidates(daily, a["window"], a["min_ratio"], a["ratio_months"])
        n = len(cands)
        ev = f"{'~'.join(a['window'])} 전후 {a['ratio_months']}개월 평균비 >= {a['min_ratio']} 고유 법정동 {n}곳 (set 집계)"
        if n < K["es_min_regions"]:
            add(2, "활성화 후보 지역 수", "실패", ev, f"{K['es_min_regions']}곳 미만 — 이벤트 스터디 포기")
        elif n < K["cf_min_regions"]:
            add(2, "활성화 후보 지역 수", "경고", ev, f"{K['cf_min_regions']}곳 미만 — Causal Forest 포기, 2x2 서브그룹")
        else:
            add(2, "활성화 후보 지역 수", "통과", ev)
    guarded(2, "활성화 후보 지역 수", check2)

    # 3·4·6. SHC002 표본
    shc_state = {}

    def load_shc():
        scan = identification.scan_shc002(paths["shc002"], C, ind, P["shc"], nrows=K["sample_rows"], extra_stats=True)
        shc_state["stats"] = scan["stats"]

    def check3():
        load_shc()
        vals = shc_state["stats"]["tizo_values"]
        ev = f"TIZO_CLCD {len(vals)}개: {vals} (앞 {K['sample_rows']:,}행)"
        if len(vals) < K["min_tizo"]:
            add(3, "시간대구간 수", "경고", ev, f"{K['min_tizo']}개 미만 — 시간대 축 포기")
        else:
            add(3, "시간대구간 수", "통과", ev)
    guarded(3, "시간대구간 수", check3)

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
    guarded(4, "유입거리구간 정의", check4)

    def check5():
        if "fail" not in kepco_state:
            raise RuntimeError("KEPCO 로드 실패(2번 참조)")
        fail = kepco_state["fail"]
        ev = f"KEPCO 지역 {kepco_state['n_regions']}곳 중 미매칭 {fail:.1%} (예: " + \
            ", ".join(kepco_state["unmatched"][["sido", "sigungu", "emd"]].head(3).agg(" ".join, axis=1)) + ")"
        if fail >= P["bjd"]["fail_warn_rate"]:
            add(5, "한전 읍면동 매핑 실패율", "실패", ev, "행정동 기준일 가능성 — 행정동↔법정동 매핑표 필요")
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
    guarded(6, "마스킹 비율", check6)

    # 7. KEPCO_001/002 포함관계
    def check7():
        kp = dict(P["kepco"], chunksize=K["compare_rows"])
        keys = ["sido", "sigungu", "emd", "mi", "hour"]
        a = kepco_loader.load_kepco_monthly_hour(paths["kepco_001"], C["kepco"], kp, "KEPCO_001", max_chunks=1)
        b = kepco_loader.load_kepco_monthly_hour(paths["kepco_002"], C["kepco_cpo"], kp, "KEPCO_002", max_chunks=1)
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
        can = can_loader.load_can_m(paths["can_m"], C, P["can"], nrows=K["sample_rows"])
        verdict, evidence = can_loader.verify_join_key(can, P["can"])
        ev = " · ".join(f"{r['항목']}={r['값']}" for _, r in evidence.iterrows() if r["항목"] != "판정")
        if verdict == "individual":
            add(8, "CAN verify_join_key", "통과", f"individual | {ev}", "경로 A(개별 차량 클러스터링)")
        else:
            add(8, "CAN verify_join_key", "경고", f"{verdict} | {ev}", "경로 B(법정동 밀집도, 보조 근거)")
    guarded(8, "CAN verify_join_key", check8)

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

    table = pd.DataFrame(rows).sort_values("번호").reset_index(drop=True)
    if write:
        writer = OutputWriter(out_dir, **P["outputs"])
        writer.table(table, "k_kill_criteria", f"킬 크라이테리아 {len(table)}항목 판정")
    return table


def main():
    parser = argparse.ArgumentParser(description="킬 크라이테리아 자동 판정 (ESS 활성화 시 9항목)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    table = run_checks(args.config)
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
