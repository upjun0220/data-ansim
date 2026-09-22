"""반입 2 데이터 생성 — 7.csv(전기차 등록 월별 이력) · 8.csv(행정동→법정동 대응표).

개발 PC 전용(pandas · openpyxl · pyshp · pyproj · shapely). 입력은 data/ref/ 에 받아 둔 공개 자료.
  - ev/z8/**.xlsx, ev/f*.bin : 서울 열린데이터광장 OA-21236 서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동) 월별 xlsx
  - hdong_20260701.geojson   : vuski/admdongkor 행정동 경계(2026-07-01, WGS84, adm_cd2 = 행정동코드 10자리)
  - emd_20230729/emd.*       : GIS Developer 읍면동 경계(UTM-K) — 법정동 면적 겹침 계산용
출력: dist/import2/7.csv · 8.csv

8.csv 가중치 = 행정동 면적 중 해당 법정동과 겹치는 면적 비율(면적 근사). README 는 '그 행정동 전기차 중 법정동 몫'을 요구하지만
행정동 안 전기차의 법정동별 분포 자료가 없어 면적으로 대신한다.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "ref"
OUT = ROOT / "dist" / "import2"


def hdong_table() -> pd.DataFrame:
    g = json.load(open(REF / "hdong_20260701.geojson", encoding="utf-8"))
    rows = []
    for f in g["features"]:
        p = f["properties"]
        if p["sido"] != "11":
            continue
        rows.append({"code": p["adm_cd2"], "sgg": p["sggnm"], "dong": p["adm_nm"].split()[-1], "geom": shape(f["geometry"])})
    return pd.DataFrame(rows)


def read_ev_xlsx(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    ym = None
    for _, r in raw.head(12).iterrows():
        for i, v in enumerate(r):
            if isinstance(v, str) and v.strip().startswith("기준일자"):
                m = re.search(r"(\d{4})\.?(\d{2})", " ".join(str(x) for x in r.tolist() if pd.notna(x)).replace("기준일자", ""))
                ym = m.group(1) + m.group(2) if m else None
    hdr = next(i for i in range(15) if "연료" in [str(x).strip() for x in raw.iloc[i].tolist()])
    cols = [str(x).strip() for x in raw.iloc[hdr].tolist()]
    fuel_i, tot_i = cols.index("연료"), cols.index("계")
    sgg_i = 0
    dong_i = fuel_i - 1
    sgg = raw.iloc[hdr + 1:, sgg_i].ffill().astype(str)
    sgg = sgg.str.replace("서울특별시", "").str.strip()
    dong = raw.iloc[hdr + 1:, dong_i].ffill().astype(str).str.split().str[-1]
    out = pd.DataFrame({"ym": ym, "sgg": sgg, "dong": dong, "fuel": raw.iloc[hdr + 1:, fuel_i], "n": pd.to_numeric(raw.iloc[hdr + 1:, tot_i], errors="coerce")})
    return out.dropna(subset=["fuel", "n"])


def build_7(hd: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob.glob(str(REF / "ev" / "z8" / "*.xlsx")) + glob.glob(str(REF / "ev" / "f*.bin")))
    frames = []
    for f in files:
        try:
            d = read_ev_xlsx(f)
        except Exception as e:  # noqa: BLE001
            print("건너뜀", Path(f).name, e)
            continue
        if d.empty or d["ym"].isna().all():
            print("기준월 없음", Path(f).name)
            continue
        frames.append(d)
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[~ev["dong"].isin(["기타", "소계", "합계", "nan"]) & ~ev["sgg"].isin(["소계", "합계", "nan"])]
    key = hd.assign(k=hd["sgg"].str.strip() + " " + hd["dong"].str.strip()).drop_duplicates("k").set_index("k")["code"]
    ev["k"] = ev["sgg"].str.split().str[-1] + " " + ev["dong"]
    ev["행정동코드"] = ev["k"].map(key)
    miss = ev[ev["행정동코드"].isna()]
    print("행정동 이름 미매칭 행 비율: %.2f%%" % (100 * len(miss) / len(ev)), sorted(miss["k"].unique())[:10])
    ev = ev.dropna(subset=["행정동코드"])
    res = ev.groupby(["ym", "행정동코드", "fuel"], as_index=False)["n"].sum()
    res.columns = ["기준년월", "행정동코드", "연료", "대수"]
    res["대수"] = res["대수"].astype(int)
    return res.sort_values(["기준년월", "행정동코드", "연료"])


def build_8(hd: pd.DataFrame) -> pd.DataFrame:
    tr = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    sf = shapefile.Reader(str(REF / "emd_20230729" / "emd"), encoding="cp949")
    bj = []
    for sr in sf.iterShapeRecords():
        code = str(sr.record["EMD_CD"])
        if code.startswith("11"):
            bj.append((code, transform(tr.transform, shape(sr.shape.__geo_interface__))))
    tree = STRtree([g for _, g in bj])
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform
    rows = []
    for _, h in hd.iterrows():
        hg = transform(to_m, h["geom"])
        area = hg.area
        parts = []
        for i in tree.query(h["geom"]):
            inter = hg.intersection(transform(to_m, bj[i][1])).area
            if inter > 0:
                parts.append((bj[i][0], inter / area))
        tot = sum(w for _, w in parts)
        for c, w in parts:
            if w / tot >= 0.01:
                rows.append({"행정동코드": h["code"], "법정동코드": c + "00", "가중치": w / tot})
    res = pd.DataFrame(rows)
    res["가중치"] = res["가중치"] / res.groupby("행정동코드")["가중치"].transform("sum")
    return res.round({"가중치": 4})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hd = hdong_table()
    print("서울 행정동", len(hd))
    e = build_7(hd)
    e.to_csv(OUT / "7.csv", index=False, encoding="utf-8-sig")
    print("7.csv", e.shape, e["기준년월"].min(), e["기준년월"].max(), "월 수", e["기준년월"].nunique())
    h = build_8(hd)
    h.to_csv(OUT / "8.csv", index=False, encoding="utf-8-sig")
    print("8.csv", h.shape, "행정동", h["행정동코드"].nunique())
