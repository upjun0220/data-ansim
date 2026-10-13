"""3차 방문 한 줄 실행 — 센터 반입 이름 field12.py(2026-10-13).

field11 등 19개는 이미 반입했고 같은 이름은 다시 반입할 수 없어, 손 입력을 줄이는 go() 만 이 파일 하나로 더 반입한다.
이 파일은 이미 반입된 field11(run_full·run_seasons)·common11·canloader11 을 불러다 쓰기만 한다(분석 동작은 field11 과 같다).

센터(커널 재시작 후, 경로는 Copy Path 붙여 넣기):
    import sys, os; sys.path.insert(0, os.path.dirname(r"field12.py의 Copy Path"))
    from field12 import go, run_seasons
    r = go(r"KEPCO_001 CSV의 Copy Path", r"5.csv의 Copy Path", r"CAN CSV의 Copy Path")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canloader import CAN_COLUMN_ALIASES
from common import detect_columns, resolve_csv_path
from fieldday1 import run_full, run_seasons

__all__ = ["go", "run_seasons", "CAN_EXPORT"]

CAN_EXPORT = ("s3_can_join_key", "s3_can_progress", "s3_can_session_summary", "s3_can_session_dist", "s3_can_session_hist",
              "s3_can_session_by_type", "s3_can_region", "s3_can_kepco_check", "s3_can_kepco_shape", "s3_can_flex_bands",
              "s3_can_flex_summary", "s3_can_away_summary", "s3_can_away_region", "s8c_can_away_check",
              "s3_can_spatial_agreement")


def go(kepco_001_path, import_dir, can_path, seasons=False, max_rows=None, out_dir="canout", bjd_path=None):
    """3차 방문 한 줄 실행(손 입력을 줄이려고): CAN 진단 → CAN만 실행 → 판정 한 줄 → 반출 신청 목록.
    seasons=True 면 이어서 계절 강건성(약 45~50분). 메모리가 모자라면 max_rows=5000000(CAN 차량 표본 행 수 상한).
    화면에는 열 이름·값의 종류·시각 형식만 보인다(원본 행을 적지 않는다)."""
    import re

    from common import read_columns

    can = resolve_csv_path(can_path, "TB_TBE_TERMINAL_LOGMOCEAN")
    m = detect_columns(can, CAN_COLUMN_ALIASES, "TB_TBE_TERMINAL_LOGMOCEAN", required=set(CAN_COLUMN_ALIASES))
    head = read_columns(can, {"g": m["charging"], "t": m["time"]}, "CAN", nrows=200000)
    print("[1] CAN 열:", m)
    print("    충전 여부 값 종류:", head["g"].value_counts().head(10).to_dict())
    print("    시각 형식(앞 2개):", head["t"].head(2).tolist())

    extra = {"stages": {"can_only": True}, **({"can": {"max_rows": int(max_rows)}} if max_rows else {})}
    r = run_full(kepco_001_path, import_dir, can_path, bjd_path, out_dir=out_dir, extra_params=extra)
    print("[2] 단계 상태:")
    print(r["stages"][["단계", "이름", "상태", "비고"]].to_string(index=False))

    csv = Path(r["out_dir"]) / "csv"
    prog = csv / "s3_can_progress.csv"
    if prog.is_file():
        p = pd.read_csv(prog, encoding="utf-8-sig").set_index("항목")["값"]
        print("[3] CAN:", " · ".join(f"{k} {p[k]}" for k in ("마지막 통과 단계", "판정", "식별번호 수", "세션 수", "읽기 범위") if k in p))
    chk = csv / "s8c_can_away_check.csv"
    if chk.is_file():
        c = pd.read_csv(chk, encoding="utf-8-sig").set_index("항목")["값"].astype(str)
        n = re.match(r"\d+", c.get("비교 법정동 수", "0"))
        n = int(n.group()) if n else 0
        key = next((k for k in c.index if k.startswith("순위 상관(원정 충전 세션 비율")), None)
        num = re.match(r"-?\d+\.\d+", c[key]) if key else None
        rho = float(num.group()) if num else np.nan
        ok = np.isfinite(rho) and rho <= -0.2 and n >= 30
        print(f"    원정 충전 ↔ 접근성 순위 상관 {rho:.3f} · 동네 {n}곳 → 사전 기준(≤ −0.2, 30곳 이상) "
              f"{'충족 — 발표 본문' if ok else '미충족 — 한 줄로만'}")
    else:
        print("    원정 충전 검증 없음(개별 차량 판정이 아니거나 CAN 단계 실패) — s3_can_progress 를 반출")
    png = Path(r["out_dir"]) / "png"
    print("[4] 반출 신청(PNG):", ", ".join(n for n in CAN_EXPORT if (png / f"{n}.png").is_file()) or "없음")
    out = {"can": r}
    if seasons:
        s = run_seasons(kepco_001_path, import_dir, bjd_path)
        print("[5] 계절 강건성:")
        print(s["robustness"].to_string(index=False))
        heads = sorted(Path(s["out_dir"]).glob("window_*/png/s8a_plus_headline.png"))
        print("    반출 신청: s9_season_robustness, s9_season_stable +",
              ", ".join(f"{h.parent.parent.name}/s8a_plus_headline" for h in heads))
        out["seasons"] = s
    return out
