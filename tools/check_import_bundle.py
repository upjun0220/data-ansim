"""반입 묶음 점검 — 반입 가능한 것은 .py(코드)·.csv/.txt(데이터·설정)·.ttf 등 폰트뿐이다(zip·json·md 불가, 총 50MB, 파일명 영문·숫자만).
파일 개수 제한은 두지 않는다(센터 규정 확인 결과 "최대 10개"는 아님). 폴더에 checksums.txt 가 있으면 해시도 대조한다.

사용: python tools/check_import_bundle.py <폴더 또는 파일> [...]   (위반이 있으면 종료코드 1)

v11 구성: 코드 모듈 21개(.py) + fieldtemplate.txt·holidays.txt·requirements.txt·README.txt·checksums.txt
+ 데이터 2.csv~9.csv + (선택) 10.ttf. 데이터 이름은 tools/make_import_bundle.py 의 IMPORT_NAMES, 이전 반입 이름과 겹치면 위반.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
FONT_EXT = {".ttf", ".otf", ".ttc"}
# 반입 포털 규칙: 파일명에는 영문 대·소문자와 숫자만(공백·한글·_·- 불가). 확장자 앞의 점 하나만 허용한다.
SAFE_NAME = re.compile(r"^[A-Za-z0-9]+\.[A-Za-z0-9]+$")


# 이전에 반입했거나 반입 요청한 파일명(2026-09-21 V10 반입). 새 반입에 같은 이름을 쓰지 않는다.
PREVIOUS_NAMES = {"dataansimbundle.py", "dataansimcodev10.zip", "bjdmaster.csv", "bjdcrosswalk.csv",
                  "bjdcentroids.csv", "smp.csv", "evstations.csv", "evregistration.csv", "koreanfont.ttf",
                  # 9/22 데이터 반입, 이후 코드 반입·신청(같은 이름 재사용 불가)
                  *(f"{i}.csv" for i in range(2, 10)), "10.ttf", "day1bundle.py",
                  *(f"day1bundle{i}.py" for i in range(2, 9)), "ansim9.py", "ansim10.py",
                  # 9/22·9/23 반입 zip 안의 코드·설정 파일(모듈 원래 이름 — 그래서 v12.1 개별 반입은 이름 끝에 버전 번호)
                  "bjdmapping.py", "canloader.py", "common.py", "config.py", "diagnostics.py", "equityaccess.py",
                  "essoptimizer.py", "headline.py", "heterogeneity.py", "identification.py", "kep007loader.py",
                  "kepcoloader.py", "killcriteria.py", "loadaxis.py", "loadforecast.py", "loadscenario.py", "outputs.py",
                  "pipeline.py", "priorityscore.py", "weatherloader.py", "holidays.txt", "checksums.txt", "fieldtemplate.txt",
                  "requirements.txt", "README.txt", "README1.txt", "BjdMapping.py", "CanLoader.py", "Common.py", "Config.py",
                  "Diagnostics.py", "EssOptimizer.py", "Heterogeneity.py", "Identification.py", "Kep007Loader.py",
                  "KepcoLoader.py", "KillCriteria.py", "LoadAxis.py", "LoadForecast.py", "Outputs.py", "Pipeline.py",
                  "FieldTemplate.txt", "FileNameMap.txt", "IndustryCodesTemplate.txt", "ReferenceGuide.txt",
                  "Requirements.txt", "SHA256SUMS.txt", "BjdMaster20260729.txt", "BjdCrosswalk20251231.csv",
                  "CentroidMasterCheck.csv", "EmdCentroids20230729.csv"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    return files


def kind(f):
    ext = f.suffix.lower()
    return "code" if ext == ".py" else "data" if ext in {".csv", ".txt"} else "font" if ext in FONT_EXT else "other"


def check(files):
    """(행 목록, 위반 목록)."""
    rows = [(f.name, kind(f), f.stat().st_size) for f in files]
    total = sum(r[2] for r in rows)
    kinds = [r[1] for r in rows]
    problems = []
    if total > MAX_BYTES:
        problems.append(f"총 {total / 1024 ** 2:.1f}MB > 최대 {MAX_BYTES // 1024 ** 2}MB")
    if "code" not in kinds:
        problems.append("파이썬(.py) 파일이 없음")
    problems += [f"허용되지 않는 형식(.py·.csv·.txt·폰트만 가능): {n}" for n, k, _ in rows if k == "other"]
    problems += [f"파일명 규칙 위반(영문·숫자만): {n}" for n, _, _ in rows if not SAFE_NAME.match(n)]
    problems += [f"이전 반입 파일명과 중복: {n}" for n, _, _ in rows if n.lower() in {p.lower() for p in PREVIOUS_NAMES}]
    for f in files:
        if kind(f) == "code":
            try:
                ast.parse(f.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                problems.append(f"{f.name} 은(는) 올바른 파이썬 파일이 아님: {type(exc).__name__}")
    for sums in (f for f in files if f.name == "checksums.txt"):
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            target = sums.parent / name
            if not target.exists():
                problems.append(f"checksums.txt 에 있는데 없음: {name}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                problems.append(f"해시 불일치(옛 버전·전송 오류 의심): {name}")
    return rows, problems


def main(argv):
    if not argv:
        sys.exit(__doc__)
    rows, problems = check(collect(argv))
    print(f"{'파일':<40}{'구분':<8}{'MB':>8}")
    for name, k, size in rows:
        print(f"{name:<40}{k:<8}{size / 1024 ** 2:>8.2f}")
    kinds = [r[1] for r in rows]
    print(f"\n합계 {len(rows)}개 · {sum(r[2] for r in rows) / 1024 ** 2:.1f}MB · 코드(.py) {kinds.count('code')} · 데이터(.csv/.txt) {kinds.count('data')} · 폰트 {kinds.count('font')}")
    print("판정: " + ("통과" if not problems else "위반 — " + "; ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
