"""공휴일 달력 받기 — 한국천문연구원 특일 정보 API(공공데이터포털 15012690, getRestDeInfo) → config/holidays.yaml.

인터넷이 되는 곳에서 반입 전에 한 번 돌린다. 안심구역 파이프라인은 이 스크립트를 호출하지 않는다.
서비스키는 환경변수 DATAGOKR_SERVICE_KEY 로만 받는다(코드·설정 파일에 적지 않는다).

    set DATAGOKR_SERVICE_KEY=...          (PowerShell: $env:DATAGOKR_SERVICE_KEY="...")
    python tools/fetch_holidays.py 2024 2025 [--out config/holidays.yaml]

저장 형식은 JSON 문법의 YAML(표준 json 으로 읽힘). 유형(holiday·long_holiday·pre_post_holiday)은 받은 공휴일 날짜로만
나누며 휴일을 규칙으로 만들지 않는다. 대체공휴일·임시공휴일·선거일이 빠졌으면 관보·정부 발표로 직접 추가하고
source 에 출처를 적는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weatherloader import classify_holidays, dump_holidays  # noqa: E402

URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
SOURCE = "한국천문연구원 특일 정보 API(getRestDeInfo)"


def fetch_year(year, key):
    query = urllib.parse.urlencode({"solYear": year, "numOfRows": 100, "_type": "json"})
    with urllib.request.urlopen(f"{URL}?ServiceKey={key}&{query}", timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    header = body["response"]["header"]
    if header.get("resultCode") != "00":
        raise RuntimeError(f"{year}: API 오류 {header}")
    items = (body["response"]["body"].get("items") or {}).get("item", [])
    items = [items] if isinstance(items, dict) else items
    return {f"{str(i['locdate'])[:4]}-{str(i['locdate'])[4:6]}-{str(i['locdate'])[6:8]}":
            {"name": i["dateName"], "public_holiday": i.get("isHoliday") == "Y"} for i in items}


def build(raw, long_min_days=3, count_weekends=True):
    public = [d for d, e in raw.items() if e["public_holiday"]]
    types = classify_holidays(public, long_min_days, count_weekends)
    return {d: {"type": t, "public_holiday": raw.get(d, {}).get("public_holiday", False),
                "name": raw.get(d, {}).get("name", "연휴 전날·다음날"),
                "source": SOURCE if d in raw else f"{SOURCE} 공휴일로 유형 구분"} for d, t in types.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("years", nargs="+", type=int)
    parser.add_argument("--out", default=str(ROOT / "config" / "holidays.yaml"))
    parser.add_argument("--long-min-days", type=int, default=3)
    args = parser.parse_args()
    key = os.environ.get("DATAGOKR_SERVICE_KEY")
    if not key:
        sys.exit("환경변수 DATAGOKR_SERVICE_KEY 가 없음")
    raw = {}
    for year in args.years:
        raw.update(fetch_year(year, key))
    entries = build(raw, args.long_min_days)
    Path(args.out).write_text(dump_holidays(entries), encoding="utf-8")
    print(f"{args.out}: {len(entries)}일 (공휴일 {sum(e['public_holiday'] for e in entries.values())}) — "
          "대체·임시공휴일·선거일 포함 여부를 관보로 확인할 것")


if __name__ == "__main__":
    main()
