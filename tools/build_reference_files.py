"""현장 참고자료 생성 — 법정동코드 마스터 · 법정동 코드대응 · 법정동 중심점 CSV.

개발 PC 전용(pyshp · pyproj 필요). 현장 파이프라인 코드는 이 패키지들을 쓰지 않는다 — 산출물 파일만 반입한다.

실행(PowerShell, 저장소 루트):
    .venv\\Scripts\\python.exe -m pip install pyshp pyproj
    .venv\\Scripts\\python.exe tools\\build_reference_files.py

입력(공개 데이터, data/ref/ 에 받아 둔다):
  - 국토교통부_전국 법정동 CSV — 공공데이터포털 15063424 (법정동코드·시도명·시군구명·읍면동명·리명·순번·생성일자)
    ⚠ 현존 코드만 들어 있다(폐지 코드 없음)
  - 읍면동 경계 SHP — GIS Developer emd_YYYYMMDD.zip (EMD_CD 8자리 = 법정동코드 앞 8자리, .prj 없음)
    ⚠ [9/16 확인] 좌표 범위 x 746k~1,388k · y 1,458k~2,068k → UTM-K(EPSG:5179) 로 판단

기준코드 원칙
  분석 데이터(KEPCO 2023~2025 · SHC 2025)의 끝인 --as-of(기본 2025-12-31)에 유효한 코드를 기준코드로 삼는다.
  경계 파일(2023)과 현행 마스터(2026)를 읍면동명으로 대응시켜 개편 전/후 코드 쌍을 만들고,
    변경일 ≤ as-of → 신코드가 기준 (예: 2023-12-29 부천 3구 신설)
    변경일 >  as-of → 구코드가 기준 (예: 2026-06-30 전남광주통합특별시·인천 행정체제 개편, 2026-02 화성 4구)
  마스터에는 신·구 코드를 모두 넣어 원천 텍스트가 어느 시점 명칭이든 매칭되게 하고,
  파이프라인이 코드대응표로 기준코드 하나로 묶는다(시계열 단절·SHC 조인 끊김 방지).
  ⚠ [9/16 확인] 경계 2023-07 ↔ 마스터 2026-07 사이 개편: 전남광주 623 · 인천 중·동·서구 80 · 화성 37 · 부천 24 · 기타 4

출력(dist/field_refs/) — 반입 포털은 파일명에 영문 대소문자·숫자만 허용한다(공백·한글·_·- 불가). 그래서 이름을 고정한다:
  bjdmaster.csv      — 법정동코드 마스터(법정동코드 · 법정동명 · 폐지여부, 쉼표)   → paths.bjd_master
  bjdcrosswalk.csv   — 코드 · 기준코드 (+ 구코드 · 신코드 · 변경일 · 명칭)            → paths.bjd_crosswalk
  bjdcentroids.csv   — 법정동코드(기준코드) · 위도 · 경도(WGS84) · 산출방식 · 코드보정 → paths.emd_centroids
  centroiddiff.csv   — 중심점 없는 기준코드 · 대응 실패 코드 (반입하지 않는 확인용)
  refguide.txt       — 출처 · 기준일 · 한계 (반입하지 않는 확인용)
반입은 위 CSV 3개를 각각 올린다(zip은 코드만). 기준일은 refguide.txt 에 적는다.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bjdmapping import load_bjd_master, load_crosswalk, load_emd_centroids  # noqa: E402
from config import DEFAULT_COLUMNS  # noqa: E402

# 경계 파일의 옛 시도코드 → 현행. 앞자리만 바뀌고 나머지 자리가 같은 경우(마스터에 그 코드가 있을 때만 적용)
#   전북특별자치도 출범 2024-01-18: 45 → 52 / 군위군 대구 편입 2023-07-01: 47720 → 27720
CODE_REMAP = [("47720", "27720"), ("45", "52")]
# 통합으로 시도코드 자체가 사라진 경우 — 구 시도명과 신 시도코드
OLD_SIDO_NAMES = {"29": "광주광역시", "46": "전라남도"}
MERGED_PREFIX = {"29": "12", "46": "12"}
# 현행 마스터에 이름이 남아 있지 않은 폐지 시군구(2026-06-30 인천 행정체제 개편). 이 표에 없는 폐지 시군구가
# 나오면 멈춘다 — 추측으로 이름을 채우지 않는다.
ABOLISHED_SGG_NAMES = {"28110": "중구", "28140": "동구", "28260": "서구"}


def read_master(src):
    df = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    need = ["법정동코드", "시도명", "시군구명", "읍면동명", "리명", "생성일자"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"국토교통부 법정동 CSV 컬럼 없음: {missing} / 실제: {list(df.columns)}")
    df = df.dropna(subset=["법정동코드"]).copy()
    df["법정동코드"] = df["법정동코드"].str.strip()
    bad = df["법정동코드"].str.len() != 10
    if bad.any():
        raise ValueError(f"10자리가 아닌 법정동코드 {int(bad.sum())}건: {df.loc[bad, '법정동코드'].head().tolist()}")
    for c in ("시도명", "시군구명", "읍면동명", "리명", "생성일자"):
        df[c] = df[c].fillna("").str.strip()
    return df


def emd_level(df):
    c = df["법정동코드"]
    return df[(c.str[8:] == "00") & (c.str[5:8] != "000")]


def read_shp(shp_base):
    reader = shapefile.Reader(str(shp_base), encoding="cp949")
    fields = [f[0] for f in reader.fields[1:]]
    for f in ("EMD_CD", "EMD_KOR_NM"):
        if f not in fields:
            raise KeyError(f"SHP 에 {f} 없음: {fields}")
    x0, y0, x1, y1 = reader.bbox
    if not (500_000 < x0 < x1 < 1_500_000 and 1_300_000 < y0 < y1 < 2_200_000):
        raise ValueError(f"좌표 범위 {reader.bbox} 가 UTM-K(EPSG:5179) 로 보이지 않음 — 좌표계 확인 필요")
    return reader, fields.index("EMD_CD"), fields.index("EMD_KOR_NM")


def remap_code(code10, master_codes):
    if code10 in master_codes:
        return code10, ""
    for old, new in CODE_REMAP:
        cand = new + code10[len(old):]
        if code10.startswith(old) and cand in master_codes:
            return cand, f"{old}→{new}"
    return code10, ""


def build_crosswalk(master, shp_items, as_of):
    """경계에만 있는 코드(개편 전) ↔ 경계에 없는 마스터 코드(개편 후)를 읍면동명으로 대응."""
    emd = emd_level(master)
    mcodes = set(emd["법정동코드"])
    shp_codes = {c for c, _ in shp_items}
    old = [(c, nm) for c, nm in shp_items if c not in mcodes]
    pool = emd[~emd["법정동코드"].isin(shp_codes)]
    pool_names = {sgg: set(g["읍면동명"]) for sgg, g in pool.groupby(pool["법정동코드"].str[:5])}
    old_names = {}
    for c, nm in old:
        old_names.setdefault(c[:5], set()).add(nm)
    sido_name = dict(zip(master.loc[master["법정동코드"].str[2:] == "00000000", "법정동코드"].str[:2],
                         master.loc[master["법정동코드"].str[2:] == "00000000", "시도명"]))
    sido_name.update(OLD_SIDO_NAMES)
    is_sgg = (master["법정동코드"].str[5:] == "00000") & (master["법정동코드"].str[2:5] != "000")
    sgg_name = dict(zip(master.loc[is_sgg, "법정동코드"].str[:5], master.loc[is_sgg, "시군구명"]))
    info = pool.set_index("법정동코드")

    rows, fails = [], []
    for c, nm in old:
        cands = pool[(pool["읍면동명"] == nm) & (
            (pool["법정동코드"].str[:2] == c[:2]) | (pool["법정동코드"].str[:2] == MERGED_PREFIX.get(c[:2], "")))]
        if len(cands) > 1:
            score = Counter({code: len(old_names[c[:5]] & pool_names[code[:5]]) for code in cands["법정동코드"]})
            top = score.most_common(2)
            cands = cands[cands["법정동코드"] == top[0][0]] if len(top) == 1 or top[0][1] > top[1][1] else cands.iloc[0:0]
        if len(cands) != 1:
            fails.append({"구분": "코드대응 실패", "법정동코드": c, "이름_또는_경계코드": nm})
            continue
        new = cands["법정동코드"].iloc[0]
        # 구 시군구명: 폐지 시군구 표 → 현행 마스터에 같은 시군구코드가 남아 있으면 그 이름 →
        #   시도 통합(29·46→12)은 시군구 이름이 그대로라 신코드의 시군구명(동구→동구). 그 밖은 추측하지 않고 멈춘다
        old_sgg = ABOLISHED_SGG_NAMES.get(c[:5]) or sgg_name.get(c[:5]) or \
            (info.at[new, "시군구명"] if c[:2] in MERGED_PREFIX else None)
        if old_sgg is None:
            raise ValueError(f"구 시군구명 미상: {c[:5]} (경계 {c} {nm}) — ABOLISHED_SGG_NAMES 에 근거와 함께 추가할 것")
        changed = info.at[new, "생성일자"]
        rows.append({"구코드": c, "신코드": new, "변경일": changed, "구_시도명": sido_name[c[:2]], "구_시군구명": old_sgg,
                     "신_시도명": info.at[new, "시도명"], "신_시군구명": info.at[new, "시군구명"], "읍면동명": nm})
    cw = pd.DataFrame(rows)
    cw["기준"] = np.where(cw["변경일"] <= as_of, "신코드", "구코드")
    cw["기준코드"] = np.where(cw["기준"] == "신코드", cw["신코드"], cw["구코드"])
    cw["코드"] = np.where(cw["기준"] == "신코드", cw["구코드"], cw["신코드"])
    multi = cw[cw["코드"].duplicated(keep=False)]
    if not multi.empty:
        raise ValueError(f"한 코드가 여러 기준코드로 대응됨(동 통합 추정): {multi[['코드', '기준코드']].head().values.tolist()}")
    return cw, pd.DataFrame(fails, columns=["구분", "법정동코드", "이름_또는_경계코드"])


def build_master(master, cw, out_txt):
    names = master[["시도명", "시군구명", "읍면동명", "리명"]].apply(lambda r: " ".join(v for v in r if v), axis=1)
    out = pd.DataFrame({"법정동코드": master["법정동코드"], "법정동명": names, "폐지여부": "존재"})
    have = set(out["법정동코드"])
    extra = []
    for sd in sorted(set(cw["구코드"].str[:2])):
        if f"{sd}00000000" not in have:
            extra.append({"법정동코드": f"{sd}00000000", "법정동명": cw.loc[cw["구코드"].str[:2] == sd, "구_시도명"].iloc[0]})
    for sgg5, g in cw.groupby(cw["구코드"].str[:5]):
        if f"{sgg5}00000" not in have:
            extra.append({"법정동코드": f"{sgg5}00000", "법정동명": f"{g['구_시도명'].iloc[0]} {g['구_시군구명'].iloc[0]}"})
    extra += [{"법정동코드": r.구코드, "법정동명": f"{r.구_시도명} {r.구_시군구명} {r.읍면동명}"} for r in cw.itertuples()]
    out = pd.concat([out, pd.DataFrame(extra).assign(폐지여부="존재")], ignore_index=True)
    if out["법정동코드"].duplicated().any():
        raise ValueError(f"복원한 구코드가 현행 코드와 겹침: {out.loc[out['법정동코드'].duplicated(), '법정동코드'].head().tolist()}")
    out.to_csv(out_txt, index=False, encoding="utf-8-sig")
    return out, len(extra)


def _ring_centroid(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if a == 0:
        return 0.0, 0.0, 0.0
    return a, ((x + x1) * cross).sum() / (6 * a), ((y + y1) * cross).sum() / (6 * a)


def _inside(px, py, rings):
    """짝홀 규칙 — 구멍(내부 링)도 자연히 처리된다."""
    c = False
    for r in rings:
        x, y = r[:, 0], r[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        m = (y > py) != (y1 > py)
        if m.any():
            xs = x[m] + (py - y[m]) * (x1[m] - x[m]) / (y1[m] - y[m])
            c ^= bool(np.sum(xs > px) % 2)
    return c


def _scanline_point(py, rings):
    """y=py 수평선이 도형 내부를 지나는 가장 긴 구간의 중점. 없으면 None."""
    xs = []
    for r in rings:
        x, y = r[:, 0], r[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        m = (y > py) != (y1 > py)
        xs.extend((x[m] + (py - y[m]) * (x1[m] - x[m]) / (y1[m] - y[m])).tolist())
    xs.sort()
    best = None
    for a, b in zip(xs[0::2], xs[1::2]):
        if best is None or b - a > best[1] - best[0]:
            best = (a, b)
    return None if best is None else ((best[0] + best[1]) / 2.0, py)


def polygon_point(shape):
    """면적가중 중심점. 오목한 동이라 중심점이 경계 밖이면 도형 내부 점으로 보정한다."""
    pts = np.asarray(shape.points, dtype=float)
    bounds = list(shape.parts) + [len(pts)]
    rings = [pts[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= 3]
    parts = [_ring_centroid(r) for r in rings]
    area = sum(p[0] for p in parts)
    if area == 0:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean()), "정점평균(면적0)"
    cx = sum(p[0] * p[1] for p in parts) / area
    cy = sum(p[0] * p[2] for p in parts) / area
    if _inside(cx, cy, rings):
        return cx, cy, "면적중심"
    pt = _scanline_point(cy, rings)
    if pt is None or not _inside(pt[0], pt[1], rings):
        largest = max(rings, key=lambda r: abs(_ring_centroid(r)[0]))
        pt = _scanline_point((largest[:, 1].min() + largest[:, 1].max()) / 2.0, rings)
    return (pt[0], pt[1], "내부점보정") if pt else (cx, cy, "면적중심(보정실패)")


def build_centroids(reader, i_cd, master_codes, to_canonical, out_csv):
    to_wgs = Transformer.from_crs(5179, 4326, always_xy=True)
    rows = []
    for sr in reader.iterShapeRecords():
        code, fix = remap_code(str(sr.record[i_cd]).strip() + "00", master_codes)
        canon = to_canonical.get(code, code)
        x, y, how = polygon_point(sr.shape)
        lon, lat = to_wgs.transform(x, y)
        rows.append({"법정동코드": canon, "위도": round(lat, 6), "경도": round(lon, 6), "산출방식": how,
                     "코드보정": fix + (" · 기준코드" if canon != code else "")})
    df = pd.DataFrame(rows)
    if not (df["위도"].between(33, 39).all() and df["경도"].between(124, 132.5).all()):
        raise ValueError("변환 좌표가 한반도 범위를 벗어남 — 좌표계 판단 오류")
    n_dup = int(df["법정동코드"].duplicated(keep=False).sum())
    df = df.drop_duplicates("법정동코드")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df, n_dup


def main():
    parser = argparse.ArgumentParser(description="법정동코드 마스터 · 코드대응 · 법정동 중심점 생성")
    parser.add_argument("--master", default="data/ref/bjd_molit_20260729.csv")
    parser.add_argument("--emd", default="data/ref/emd_20230729/emd")
    parser.add_argument("--master-date", default="20260729")
    parser.add_argument("--emd-date", default="20230729")
    parser.add_argument("--as-of", default="2025-12-31", help="기준코드 날짜 — 분석 데이터 기간의 끝")
    parser.add_argument("--out", default="dist/field_refs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for old_file in out.glob("*"):
        old_file.unlink()
    tag = args.as_of.replace("-", "")

    master = read_master(ROOT / args.master)
    master_emd_codes = set(emd_level(master)["법정동코드"])
    reader, i_cd, i_nm = read_shp(ROOT / args.emd)
    shp_items = []
    for r in reader.iterRecords():
        code, _ = remap_code(str(r[i_cd]).strip() + "00", master_emd_codes)
        shp_items.append((code, str(r[i_nm]).strip()))

    cw, fails = build_crosswalk(master, shp_items, args.as_of)
    cw_csv = out / "bjdcrosswalk.csv"
    cw[["코드", "기준코드", "기준", "구코드", "신코드", "변경일", "구_시도명", "구_시군구명", "신_시도명", "신_시군구명",
        "읍면동명"]].sort_values(["변경일", "구코드"]).to_csv(cw_csv, index=False, encoding="utf-8-sig")
    master_txt = out / "bjdmaster.csv"
    master_out, n_restored = build_master(master, cw, master_txt)

    to_canonical = dict(zip(cw["코드"], cw["기준코드"]))
    all_codes = master_emd_codes | set(cw["구코드"])
    cent_csv = out / "bjdcentroids.csv"
    cent, n_dup = build_centroids(reader, i_cd, all_codes, to_canonical, cent_csv)

    # 파이프라인 로더로 그대로 읽히는지 · 기준코드 기준 보유율
    mapping = load_crosswalk(cw_csv, DEFAULT_COLUMNS["crosswalk"])
    loaded_master = load_bjd_master(master_txt, DEFAULT_COLUMNS["bjd"])
    loaded_cent = load_emd_centroids(cent_csv, DEFAULT_COLUMNS["centroid"])
    canonical_set = {mapping.get(c, c) for c in loaded_master["bjd_code"]}
    have = set(loaded_cent["bjd_code"])
    coverage = len(have & canonical_set) / len(canonical_set)
    name_of = dict(zip(loaded_master["bjd_code"], loaded_master["region_name"]))
    missing = sorted(canonical_set - have)
    stray = sorted(have - canonical_set)
    diff = pd.concat([
        pd.DataFrame({"구분": "중심점 없음(기준코드)", "법정동코드": missing, "이름_또는_경계코드": [name_of.get(c, "") for c in missing]}),
        pd.DataFrame({"구분": "마스터에 없는 중심점 코드", "법정동코드": stray, "이름_또는_경계코드": ""}),
        fails,
    ], ignore_index=True)
    diff.to_csv(out / "centroiddiff.csv", index=False, encoding="utf-8-sig")

    summary = cw.groupby(["변경일", "기준", cw["구코드"].str[:5]]).agg(
        읍면동수=("구코드", "size"), 구=("구_시군구명", "first"), 신=("신_시군구명", lambda s: "·".join(sorted(set(s))))
    ).reset_index().rename(columns={"구코드": "구시군구코드"})
    how = cent["산출방식"].value_counts().to_dict()
    guide = f"""충전 리플맵 4.2 — 현장 참고자료 (config/fieldtemplate.json 이 이 파일 이름을 가리킨다)

1) bjdmaster.csv   → paths.bjd_master  (기준일 {args.master_date})
   출처: 국토교통부_전국 법정동 (공공데이터포털 https://www.data.go.kr/data/15063424/fileData.do), 기준 {args.master_date}
   현행 {len(master):,}행 + 개편 전 코드 복원 {n_restored:,}행 = {len(master_out):,}행 · 로더 기준 읍면동 {len(loaded_master):,}곳
   원천 텍스트가 개편 전 명칭이든 후 명칭이든 매칭되도록 둘 다 넣었다.

2) bjdcrosswalk.csv   → paths.bjd_crosswalk  (기준코드 기준일 {tag})
   기준코드 = {args.as_of}(분석 데이터 끝)에 유효한 코드. 파이프라인이 KEPCO·SHC001·SHC002·중심점 코드를 모두 이 코드로 묶는다.
   대응 {len(cw):,}쌍 · 대응 실패 {len(fails)}건
{summary.to_string(index=False)}

3) bjdcentroids.csv   → paths.emd_centroids  (경계 기준일 {args.emd_date})
   출처: GIS Developer 읍면동 경계 SHP emd_{args.emd_date}.zip (http://www.gisdeveloper.co.kr/?p=2332)
   좌표: UTM-K(EPSG:5179) → WGS84 · 법정동코드는 기준코드 · 산출방식 {how} · 중복 도형 {n_dup}
   기준코드 법정동 대비 중심점 보유 {coverage:.2%} (파이프라인 로더로 읽어 확인)

현장에서 확인할 것
 - 킬 크라이테리아 5번 미매칭 목록에 개편 지역(광주·전남·인천·부천·화성)이 남는지.
 - SHC 코드가 기준코드 체계와 다르면(예: 2025 자료가 이미 12 로 재코딩돼 있으면) 코드대응표가 그대로 흡수한다.

한계
 - 경계({args.emd_date})와 마스터({args.master_date}) 사이 동 이름까지 바뀐 경우는 대응하지 못한다 → centroiddiff.csv.
 - 중심점 최근접 근사다. 현장 QGIS 에서 최신 경계로 공간조인할 수 있으면 그 결과가 정확하다.
 - 오목한 동은 면적중심이 경계 밖이라 도형 내부 점으로 보정했다(산출방식=내부점보정).
 - 개편 전 코드는 리 단위를 복원하지 않았다(파이프라인은 읍면동 단위만 사용).
"""
    (out / "refguide.txt").write_text(guide, encoding="utf-8")
    print(guide)
    print("대조:", diff["구분"].value_counts().to_dict())
    print(diff.groupby("구분").head(8).to_string(index=False))
    print("출력:", out)


if __name__ == "__main__":
    main()
