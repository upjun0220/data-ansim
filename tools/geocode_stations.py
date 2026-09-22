"""서울 공용 충전소 주소 → 좌표(카카오 로컬 주소검색) → dist/import2/6.csv (위도·경도·충전기수).

개발 PC 전용. 키는 환경변수 KAKAO_REST_KEY (채팅·코드에 적지 않는다).
입력: data/ref/stations_seoul_to_geocode.csv (군구·주소·충전소명·충전기수) — 한국환경공단 충전소 위치 및 운영정보(2022-10-27)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ref" / "stations_seoul_to_geocode.csv"
CACHE = ROOT / "data" / "ref" / "stations_geocoded.csv"
OUT = ROOT / "dist" / "import2" / "6.csv"


def key() -> str:
    k = os.environ.get("KAKAO_REST_KEY")
    if not k and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            k = winreg.QueryValueEx(h, "KAKAO_REST_KEY")[0]
    if not k:
        sys.exit("KAKAO_REST_KEY 없음")
    return k


def geocode(addr: str, headers: dict) -> tuple[float, float] | None:
    q = re.sub(r"\(.*?\)", "", addr).strip()  # 괄호 안 건물명 제거
    for query in (q, re.sub(r"\s+\d+동.*$", "", q)):
        r = requests.get("https://dapi.kakao.com/v2/local/search/address.json", params={"query": query}, headers=headers, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        docs = r.json().get("documents", [])
        if docs:
            return float(docs[0]["y"]), float(docs[0]["x"])
    return None


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    headers = {"Authorization": "KakaoAK " + key()}
    src = pd.read_csv(SRC, encoding="utf-8-sig")
    done = pd.read_csv(CACHE, encoding="utf-8-sig") if CACHE.exists() else pd.DataFrame(columns=["주소", "위도", "경도"])
    got = {r["주소"]: (r["위도"], r["경도"]) for _, r in done.iterrows()}
    addrs = [a for a in src["주소"].unique() if a not in got]
    if limit:
        addrs = addrs[:limit]
    print("변환 대상", len(addrs), "이미 있음", len(got))
    for i, a in enumerate(addrs, 1):
        try:
            got[a] = geocode(a, headers) or (None, None)
        except RuntimeError as e:
            print("중단:", e)
            break
        if i % 200 == 0:
            pd.DataFrame([(k, *v) for k, v in got.items()], columns=["주소", "위도", "경도"]).to_csv(CACHE, index=False, encoding="utf-8-sig")
            print(i, "/", len(addrs))
        time.sleep(0.05)
    geo = pd.DataFrame([(k, *v) for k, v in got.items()], columns=["주소", "위도", "경도"])
    geo.to_csv(CACHE, index=False, encoding="utf-8-sig")
    m = src.merge(geo, on="주소", how="left")
    ok = m.dropna(subset=["위도"])
    ok = ok[ok["위도"].between(37.42, 37.71) & ok["경도"].between(126.76, 127.19)]  # 서울 범위 밖 제외
    print("좌표 성공 충전소 %d / %d, 충전기 %d / %d" % (len(ok), len(m), ok["충전기수"].sum(), m["충전기수"].sum()))
    if not limit:
        ok.groupby(["위도", "경도"], as_index=False)["충전기수"].sum().round({"위도": 6, "경도": 6}).to_csv(OUT, index=False, encoding="utf-8-sig")
        print("6.csv 저장")
