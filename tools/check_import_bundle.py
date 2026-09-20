"""반입 묶음 점검 — 코드 zip 1 + 데이터 CSV + 폰트가 반입 규정(최대 10개·총 50MB·CSV만)에 맞는지 표로 보인다.

사용: python tools/check_import_bundle.py <폴더 또는 파일> [...]   (위반이 있으면 종료코드 1)
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

MAX_FILES, MAX_BYTES = 10, 50 * 1024 * 1024
FONT_EXT = {".ttf", ".otf", ".ttc"}
# 반입 포털 규칙: 파일명에는 영문 대·소문자와 숫자만(공백·한글·특수문자 불가). 확장자 앞의 점 하나만 허용한다.
SAFE_NAME = re.compile(r"^[A-Za-z0-9]+\.[A-Za-z0-9]+$")
SAFE_PART = re.compile(r"^[A-Za-z0-9]+(\.[A-Za-z0-9]+)?$")  # zip 안의 폴더·파일 이름 조각


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    return files


def kind(f):
    ext = f.suffix.lower()
    return "code" if ext == ".zip" else "data" if ext == ".csv" else "font" if ext in FONT_EXT else "other"


def check(files):
    """(행 목록, 위반 목록). 폰트는 파일 수·용량에 포함하되 없어도 위반은 아니다(현장 서버 폰트 사용 가능)."""
    rows = [(f.name, kind(f), f.stat().st_size) for f in files]
    total = sum(r[2] for r in rows)
    kinds = [r[1] for r in rows]
    problems = []
    if len(rows) > MAX_FILES:
        problems.append(f"파일 {len(rows)}개 > 최대 {MAX_FILES}개")
    if total > MAX_BYTES:
        problems.append(f"총 {total / 1024 ** 2:.1f}MB > 최대 {MAX_BYTES // 1024 ** 2}MB")
    if kinds.count("code") != 1:
        problems.append(f"코드 zip {kinds.count('code')}개(정확히 1개여야 함)")
    problems += [f"허용되지 않는 형식: {n}" for n, k, _ in rows if k == "other"]
    problems += [f"파일명 규칙 위반(영문·숫자만): {n}" for n, _, _ in rows if not SAFE_NAME.match(n)]
    for f in files:  # zip 안의 이름도 같은 규칙으로 본다(포털이 안쪽 이름까지 검사하는지 확인되지 않아 보수적으로)
        if f.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(f) as z:
                    bad = [n for n in z.namelist() if not all(SAFE_PART.match(part) for part in n.rstrip("/").split("/"))]
            except zipfile.BadZipFile:
                problems.append(f"{f.name} 은(는) 올바른 zip 이 아님")
                continue
            problems += [f"{f.name} 안의 파일명 규칙 위반: {n}" for n in bad[:5]]
    return rows, problems


def main(argv):
    if not argv:
        sys.exit(__doc__)
    rows, problems = check(collect(argv))
    print(f"{'파일':<40}{'구분':<8}{'MB':>8}")
    for name, k, size in rows:
        print(f"{name:<40}{k:<8}{size / 1024 ** 2:>8.2f}")
    kinds = [r[1] for r in rows]
    print(f"\n합계 {len(rows)}개 · {sum(r[2] for r in rows) / 1024 ** 2:.1f}MB · 코드 {kinds.count('code')} · 데이터 {kinds.count('data')}"
          f" · 폰트 {kinds.count('font')}" + ("" if "font" in kinds else " (폰트 없음 — 서버 폰트 확인)"))
    print("판정: " + ("통과" if not problems else "위반 — " + "; ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main(sys.argv[1:])