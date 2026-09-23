"""안심데이터센터 현장 실행기 — Copy Path만 붙여 넣어 실행한다(센터에서는 코드를 손으로 입력하므로 인자를 최소화).

run_first_visit: 킬 크라이테리아 21개 → 법정동 매칭률·미매칭 내부 목록 → KEPCO 월별 집계(0·1단계)에서 정상 종료.
run_full:        같은 점검 후 전체 분석(AI 예측·급증 위험·CAN×KEPCO 교차검증·접근성·결합 점수·충전기 추가 실험·
                 발견 문장·동네 카드·발표 숫자)까지 실행.
run_seasons:     평가 창을 3개월씩 옮겨 전체 분석을 반복하고 상위 동네가 계절마다 유지되는지 비교.

미매칭 상세 CSV와 로그는 센터 내부 검토용이다. 반출 후보 PNG에는 원천 지역명·오류 전문·절대경로를 싣지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import kepcoloader
from bjdmapping import BJD_COLUMN_ALIASES
from canloader import CAN_COLUMN_ALIASES
from common import detect_columns, resolve_csv_path
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS, deep_merge
from killcriteria import run_checks
from pipeline import run as run_pipeline
from pipeline import run_day1 as run_pipeline_day1

# 반입 자료 이름(tools/make_import_bundle.py IMPORT_NAMES). 2.csv 는 법정동 마스터 기본값.
IMPORT_FILES = {"bjd_crosswalk": "3.csv", "emd_centroids": "4.csv", "smp": "5.csv", "access_stations": "6.csv",
                "ev_history": "7.csv", "hdong_bjd": "8.csv", "weather": "9.csv", "font": "10.ttf"}


def run_first_visit(bjd_path, can_path, kepco_001_path=None, kepco_002_path=None, out_dir="day1out"):
    """Copy Path 네 곳만 받아 첫 방문 결과를 만든다. KEPCO는 001 우선, 없으면 002 잠정."""
    if not kepco_001_path and not kepco_002_path:
        raise ValueError("KEPCO_001 또는 KEPCO_002 Copy Path 중 하나는 필요합니다")

    bjd = resolve_csv_path(bjd_path, "TB_COMM_UMD_CODE")
    can = resolve_csv_path(can_path, "TB_TBE_TERMINAL_LOGMOCEAN")
    kepco_001 = resolve_csv_path(kepco_001_path, "KEPCO_001") if kepco_001_path else None
    kepco_002 = resolve_csv_path(kepco_002_path, "KEPCO_002") if kepco_002_path else None
    source = "001" if kepco_001 else "002"

    bjd_columns = detect_columns(bjd, BJD_COLUMN_ALIASES, "TB_COMM_UMD_CODE", required={"code"})
    if "name" not in bjd_columns and not all(key in bjd_columns for key in ("sido", "sigungu", "emd")):
        raise KeyError("법정동 전체 이름 1열 또는 시도명·시군구명·읍면동명 3열을 자동 판별하지 못했습니다")
    can_columns = detect_columns(
        can, CAN_COLUMN_ALIASES, "TB_TBE_TERMINAL_LOGMOCEAN", required=set(CAN_COLUMN_ALIASES)
    )

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "day1config.json"
    config = {
        "paths": {
            "bjd_master": str(bjd),
            "kepco_001": str(kepco_001) if kepco_001 else None,
            "kepco_002": str(kepco_002) if kepco_002 else None,
            "kepco_hourly": str(kepco_001) if kepco_001 else None,
            "can_m": str(can),
            "out_dir": str(output),
        },
        "columns": {"bjd": bjd_columns, "can_m": can_columns},
        "params": {
            "stages": {"commerce": False},
            "kepco": {"source": source},
            "energy": {"enabled": True},
            "priority": {"enabled": True},
            "kill": {"kepco_max_chunks": 3},
        },
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    kill = run_checks(config_path)
    if kill["번호"].tolist() != list(range(1, 22)):
        raise RuntimeError(f"킬 크라이테리아가 21개가 아님: {kill['번호'].tolist()}")
    pipeline = run_pipeline_day1(config_path)
    return {
        "source": f"KEPCO_{source}" + (" 잠정" if source == "002" else ""),
        "killcriteria": kill,
        "stages": pipeline["stages"],
        "out_dir": str(output),
        "반출후보": [
            str(output / "kill_criteria" / "png" / "k_kill_criteria.png"),
            str(output / "png" / "s0_bjd_match_rate.png"),
            str(output / "png" / "s0_bjd_unmatched.png"),
            str(output / "png" / "s1_kepco_monthly.png"),
        ],
        "내부전용": [str(output / "csv"), str(output / "pipeline.log")],
    }


def default_evaluation_start(kepco_path, days=None):
    """평가 창 = KEPCO 수록 마지막 날까지의 마지막 days일(결과를 보기 전에 정하는 규칙)."""
    days = int(days or DEFAULT_PARAMS["energy"]["evaluation_days"])
    dates = kepcoloader.scan_observed_dates(kepco_path, DEFAULT_COLUMNS["kepco"], DEFAULT_PARAMS["kepco"])
    if dates.empty:
        raise ValueError("KEPCO 조회기간에서 일자(YYYYMMDD)를 읽지 못함 — 월별 자료면 다음날 예측(8-A)을 할 수 없음")
    last = pd.to_datetime(dates["date"].max(), format="%Y%m%d")
    return (last - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")


def run_full(kepco_001_path, import_dir, can_path=None, bjd_path=None, evaluation_start=None, out_dir="fullout",
             extra_params=None):
    """전체 분석. import_dir = 반입 자료 2~10번이 든 폴더(또는 그중 파일 하나의 Copy Path).
    없는 파일은 해당 단계를 생략하고 '입력' 표에 적는다."""
    folder = Path(str(import_dir).strip().strip("\"'")).expanduser()
    if folder.is_file():   # 반입 파일 하나(예: 5.csv)의 Copy Path 를 넣어도 그 폴더를 쓴다
        folder = folder.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"반입 자료 폴더를 찾지 못함: {str(import_dir)!r} (현재 작업 폴더: {Path.cwd()})")
    found = {key: folder / name for key, name in IMPORT_FILES.items() if (folder / name).is_file()}
    bjd = resolve_csv_path(bjd_path or folder / "2.csv", "법정동 마스터")
    kepco = resolve_csv_path(kepco_001_path, "KEPCO_001")
    can = resolve_csv_path(can_path, "TB_TBE_TERMINAL_LOGMOCEAN") if can_path else None
    here = Path(__file__).resolve().parent   # 번들: day1code/holidays.yaml · 저장소: config/holidays.yaml
    holidays = next((p for p in (here / "holidays.yaml", here / "config" / "holidays.yaml") if p.is_file()), None)

    bjd_columns = detect_columns(bjd, BJD_COLUMN_ALIASES, "법정동 마스터", required={"code"})
    columns = {"bjd": bjd_columns}
    if can:
        columns["can_m"] = detect_columns(can, CAN_COLUMN_ALIASES, "TB_TBE_TERMINAL_LOGMOCEAN",
                                          required=set(CAN_COLUMN_ALIASES))
    start = evaluation_start or default_evaluation_start(kepco)

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {"bjd_master": str(bjd), "kepco_001": str(kepco), "kepco_hourly": str(kepco),
             "can_m": str(can) if can else None, "holidays": str(holidays) if holidays else None,
             "out_dir": str(output), **{key: str(path) for key, path in found.items()}}
    config = {
        "paths": paths,
        "columns": columns,
        "params": {
            "stages": {"commerce": False},
            "kepco": {"source": "001"},
            "energy": {"enabled": True, "evaluation_start": start},
            "priority": {"enabled": True},
            "kill": {"kepco_max_chunks": 3},
        },
    }
    config["params"] = deep_merge(config["params"], extra_params or {})
    config_path = output / "fullconfig.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    used = pd.DataFrame([{"입력": key, "반입 이름": IMPORT_FILES.get(key, ""), "사용": key in found}
                         for key in IMPORT_FILES] +
                        [{"입력": "holidays", "반입 이름": "번들 내장", "사용": paths["holidays"] is not None},
                         {"입력": "can_m", "반입 이름": "센터 CAN", "사용": can is not None},
                         {"입력": "evaluation_start", "반입 이름": start, "사용": True}])
    kill = run_checks(config_path)
    pipeline = run_pipeline(config_path)
    pngs = sorted(str(p) for p in (output / "png").glob("*.png"))
    return {
        "입력": used,
        "killcriteria": kill,
        "stages": pipeline["stages"],
        "out_dir": str(output),
        "반출후보": [str(output / "kill_criteria" / "png" / "k_kill_criteria.png")] + pngs,
        "내부전용": [str(output / "csv"), str(output / "pipeline.log"), str(config_path)],
    }


def season_windows(last_start, n_windows=4, step_months=3):
    """마지막 평가 창에서 step_months 씩 앞당긴 n_windows 개 시작일(기본: 3개월 간격 4개 = 사계절)."""
    last = pd.Timestamp(last_start)
    return [(last - pd.DateOffset(months=step_months * k)).strftime("%Y-%m-%d") for k in range(n_windows)]


def compare_windows(priority_by_start, top_n):
    """창별 8-E 순위(순위 대상만)에서 상위 top_n 이 얼마나 유지되나. (창별 표, 모든 창 공통 상위 법정동 목록)."""
    tops = {s: set(df.sort_values("rank")["bjd_code"].astype(str).head(top_n)) for s, df in priority_by_start.items()}
    common = set.intersection(*tops.values()) if tops else set()
    first = next(iter(tops.values()), set())
    rows = [{"평가 시작": s, "순위 대상 수": len(priority_by_start[s]), f"상위 {top_n}": len(t),
             "첫 창과 겹침": len(t & first), "자카드(첫 창)": len(t & first) / len(t | first) if t | first else np.nan}
            for s, t in tops.items()]
    starts = list(tops)
    pairs = [len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for i, a in enumerate(starts) for b in starts[i + 1:]
             if tops[a] | tops[b]]
    rows.append({"평가 시작": "요약", "순위 대상 수": np.nan, f"상위 {top_n}": np.nan,
                 "첫 창과 겹침": len(common), "자카드(첫 창)": float(np.mean(pairs)) if pairs else np.nan})
    table = pd.DataFrame(rows)
    table.attrs["note"] = f"요약 행: 첫 창과 겹침 = 모든 창에서 상위 {top_n} 유지 동네 수, 자카드 = 창 쌍 평균"
    return table, sorted(common)


def run_seasons(kepco_001_path, import_dir, bjd_path=None, n_windows=4, step_months=3, out_dir="seasonout",
                extra_params=None):
    """계절 강건성 — 평가 창을 옮겨 가며 전체 분석을 반복하고 상위 동네가 유지되는지 비교한다.

    속도를 위해 창마다 ESS(8-B)·2030 시나리오(8-F)·CAN 은 끈다. 창 시작일 규칙은 결과를 보기 전에 정한다.
    """
    output = Path(out_dir).resolve()
    kepco = resolve_csv_path(kepco_001_path, "KEPCO_001")
    starts = season_windows(default_evaluation_start(kepco), n_windows, step_months)
    light = deep_merge(extra_params or {}, {"stages": {"ess": False, "scenario": False}})
    priority, status = {}, []
    for start in starts:
        res = run_full(kepco, import_dir, None, bjd_path, start, output / f"window_{start}", extra_params=light)
        path = Path(res["out_dir"]) / "csv" / "s8e_priority.csv"
        stage = res["stages"].set_index("단계")["상태"].to_dict()
        status.append({"평가 시작": start, "8-A": stage.get("8-A", "-"), "8-E": stage.get("8-E", "-")})
        if path.exists():
            priority[start] = pd.read_csv(path, encoding="utf-8-sig", dtype={"bjd_code": str})
    if len(priority) < 2:
        raise RuntimeError(f"8-E 순위가 나온 창이 {len(priority)}개 — 계절 비교 불가: {status}")
    top_n = int(DEFAULT_PARAMS["priority"]["sim_top_n"])
    table, common = compare_windows(priority, top_n)
    from outputs import OutputWriter

    folder = Path(str(import_dir).strip().strip("\"'"))
    font = (folder.parent if folder.is_file() else folder) / IMPORT_FILES["font"]
    writer = OutputWriter(output, **DEFAULT_PARAMS["outputs"], font_path=str(font) if font.is_file() else None)
    writer.table(table, "s9_season_robustness", f"계절 강건성 — 평가 창별 우선순위 상위 {top_n} 유지", digits=3,
                 note=table.attrs["note"] + " · 창마다 ESS·2030·CAN 생략")
    writer.table(pd.DataFrame({"bjd_code": common}), "s9_season_stable",
                 f"모든 평가 창에서 상위 {top_n}에 남은 법정동 {len(common)}곳")
    return {"windows": pd.DataFrame(status), "robustness": table, "stable": common, "out_dir": str(output)}
