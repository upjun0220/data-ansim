"""센터 밖 — 반출한 동네 상세 PNG(s9_detail_pNN)를 옮겨 적은 CSV로 전체 색칠 지도·웹 화면용 표를 만든다.

1) 쪽마다 마지막 Σ행(코드 합·열 합·행 수)으로 옮겨 적은 값을 대조한다(오타·OCR 오류를 쪽 단위로 잡는다).
2) 공개자료 값(2SFCA·충전기 1기당 EV·최근접 거리·동네 이름)은 센터와 같은 함수(equityaccess.public_access)로 반입했던
   2~9.csv 에서 다시 계산한다 — 반출하지 않아도 된다.
3) 교차 검증: 결합점수 = w_r·급증위험_정규화 + w_e·형평성_정규화 이므로, 밖에서 다시 계산한 형평성(같은 정규화)으로
   되짚은 급증위험_정규화가 반출된 급증위험과 같은 순서여야 한다(스피어만 ≈ 1). v12 기본 정규화는 백분위라 원값과
   일차식이 아니고 순서만 보존된다. 어긋나면 옮겨 적기나 입력 파일·정규화 설정이 다르다.

옮겨 적은 CSV 형식: PNG 와 같은 열 이름(bjd_code, 순위, 결합점수, …), 마지막 행이 Σ행, 빈 칸(—)은 비워 둔다.
사용: python tools/detail_from_export.py --pages <쪽 CSV 폴더> --import-dir <2~9.csv 폴더> --out app_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bjdmapping  # noqa: E402
import equityaccess  # noqa: E402
import headline  # noqa: E402
from common import keep_region  # noqa: E402
from config import DEFAULT_COLUMNS, DEFAULT_PARAMS  # noqa: E402

IMPORT = {"bjd_master": "2.csv", "bjd_crosswalk": "3.csv", "emd_centroids": "4.csv", "access_stations": "6.csv",
          "ev_history": "7.csv", "hdong_bjd": "8.csv"}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("—", "").str.strip().replace("", np.nan),
                         errors="coerce")


def check_page(page, digits=3):
    """쪽 하나(마지막 Σ행 포함)를 대조 → (데이터 행, 문제 목록)."""
    body, tail = page.iloc[:-1].copy(), page.iloc[-1]
    body["bjd_code"] = body["bjd_code"].astype(str).str.strip()
    problems = []
    want = int(str(tail["bjd_code"]).lstrip("Σ").replace(",", ""))
    if sum(int(c) for c in body["bjd_code"]) != want:
        problems.append("법정동 코드 합 불일치")
    if str(tail.get("위험상태", "")).strip() != f"{len(body)}행":
        problems.append(f"행 수 불일치({len(body)}행 대 {tail.get('위험상태')})")
    for c in body.columns:
        if c in headline.DETAIL_FLOAT_COLS or c in headline.DETAIL_INT_COLS:
            got = _num(body[c]).round(digits).sum()
            exp = float(_num(pd.Series([tail[c]])).iloc[0]) if pd.notna(tail[c]) else 0.0
            tol = 1.0 if abs(exp) >= 1000 else 10 ** -digits * len(body) / 2 + 1e-9   # 1000 이상은 정수로 보인다
            if abs(got - exp) > tol:
                problems.append(f"{c} 합 {got:.{digits}f} ≠ {exp}")
    if "묶음" in body:   # v12.1 소표본 묶기 — 대표 동 코드(빈 칸 = 묶음 아님)
        body["묶음"] = body["묶음"].astype(str).str.strip().replace({"": np.nan, "—": np.nan, "nan": np.nan})
        if sum(int(c) for c in body["묶음"].dropna()) != int(str(tail["묶음"]).lstrip("Σ").replace(",", "") or 0):
            problems.append("묶음 코드 합 불일치")
    for c in headline.DETAIL_FLOAT_COLS + headline.DETAIL_INT_COLS:
        if c in body:
            body[c] = _num(body[c])
    return body, problems


def read_pages(folder):
    rows, report = [], []
    for f in sorted(Path(folder).glob("*.csv")):
        body, problems = check_page(pd.read_csv(f, dtype=str, encoding="utf-8-sig"))
        report.append({"쪽": f.name, "행": len(body), "문제": "; ".join(problems) or "없음"})
        rows.append(body)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(report)


def public_values(import_dir, params=DEFAULT_PARAMS, columns=DEFAULT_COLUMNS, ev_registration=None):
    """반입했던 공개자료로 센터 8-C 와 같은 접근성 표 + 동네 이름. ev_registration 은 센터에서 그 파일을 썼을 때만."""
    d = Path(import_dir)
    paths = {k: str(d / v) if (d / v).is_file() else None for k, v in IMPORT.items()} | {"ev_registration": ev_registration}
    prefix = params["region"]["sido_prefix"]
    cw = bjdmapping.load_crosswalk(paths["bjd_crosswalk"], columns["crosswalk"]) if paths["bjd_crosswalk"] else None
    cent = keep_region(bjdmapping.load_emd_centroids(paths["emd_centroids"], columns["centroid"]), prefix)
    cent["bjd_code"] = bjdmapping.canonicalize(cent["bjd_code"], cw)
    cent = cent.drop_duplicates("bjd_code")
    access, _, _ = equityaccess.public_access(paths, columns, params, cent, cw)
    master = keep_region(bjdmapping.load_bjd_master(paths["bjd_master"], columns["bjd"]), prefix)
    return access.assign(bjd_code=access["bjd_code"].astype(str)), master, cent


def cross_check(detail, access, weights=(0.5, 0.5), scale="percentile"):
    """반출 결합점수에서 형평성 몫을 빼 되짚은 급증위험_정규화 ~ 반출 급증위험 의 순위 상관(스피어만)."""
    from priorityscore import SCALERS

    eq = access.assign(equity_norm=SCALERS[scale](access["access_2sfca"], reverse=True)).set_index("bjd_code")["equity_norm"]
    d = detail.assign(equity_norm=detail["bjd_code"].map(eq)).dropna(subset=["결합점수", "급증위험", "equity_norm"])
    if len(d) < 3 or d["급증위험"].nunique() < 2:
        return np.nan, len(d)
    implied = (d["결합점수"] - weights[1] * d["equity_norm"]) / weights[0]
    return float(implied.corr(d["급증위험"], method="spearman")), len(d)


def app_table(detail, access, master, cent, scale="percentile"):
    """웹 화면 한 줄 = 법정동 하나: 반출 값 + 공개 값 + 권고 초안. 좌표는 공개 중심점."""
    from priorityscore import SCALERS

    a = access.assign(equity_norm=SCALERS[scale](access["access_2sfca"], reverse=True))
    names = master.drop_duplicates("bjd_code").set_index("bjd_code")["region_name"]
    out = detail.merge(a[["bjd_code", "access_2sfca", "equity_norm", "ev_count", "chargers_assigned", "ev_per_charger",
                          "nearest_charger_km"]], on="bjd_code", how="left")
    out.insert(1, "법정동", out["bjd_code"].map(names))
    out["급증위험_정규화"] = SCALERS[scale](out["급증위험"])   # 가려진 동네를 뺀 근사(표시용)
    if "묶음" in out:   # 묶음 구성원(G)은 대표 동의 안심구역 값을 쓴다(형평성은 자기 공개 값 그대로)
        g = out["위험상태"].eq("G") & out["묶음"].notna()
        head = out.set_index("bjd_code")
        for col in headline.DETAIL_KEPCO_COLS + ["급증위험_정규화"]:
            if col in out:
                out.loc[g, col] = out.loc[g, "묶음"].map(head[col])
    out["권고(초안)"] = [headline.recommend(r, e) if pd.notna(r) and pd.notna(e) else "—"
                       for r, e in zip(out["급증위험_정규화"], out["equity_norm"])]
    c = cent.assign(bjd_code=cent["bjd_code"].astype(str)).set_index("bjd_code")
    out["lat"], out["lon"] = out["bjd_code"].map(c["lat"]), out["bjd_code"].map(c["lon"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--import-dir", required=True)
    ap.add_argument("--out", default="app_data.csv")
    args = ap.parse_args()
    detail, report = read_pages(args.pages)
    print(report.to_string(index=False))
    access, master, cent = public_values(args.import_dir)
    r2, n = cross_check(detail, access)
    print(f"교차 검증: 결합점수 ↔ (급증위험, 공개 재계산 형평성) 순위 상관 = {r2:.5f} ({n}곳)")
    app = app_table(detail, access, master, cent)
    app.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"웹 화면 표: {args.out} · {len(app)}곳")
    if (report["문제"] != "없음").any() or not (np.isnan(r2) or r2 > 0.99):
        raise SystemExit("대조 실패 — 문제 쪽을 다시 옮겨 적을 것")


if __name__ == "__main__":
    main()
