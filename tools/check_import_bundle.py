"""반입 묶음 점검 — 반입 가능한 것은 .py(코드)·.csv/.txt(데이터)·.ttf 등 폰트뿐이다(zip·json·md 불가, 최대 10개·총 50MB·파일명 영문·숫자만).

사용: python tools/check_import_bundle.py <폴더 또는 파일> [...]   (위반이 있으면 종료코드 1)
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MAX_FILES, MAX_BYTES = 10, 50 * 1024 * 1024
FONT_EXT = {".ttf", ".otf", ".ttc"}
# 반입 포털 규칙: 파일명에는 영문 대·소문자와 숫자만(공백·한글·_·- 불가). 확장자 앞의 점 하나만 허용한다.
SAFE_NAME = re.compile(r"^[A-Za-z0-9]+\.[A-Za-z0-9]+$")


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
    if len(rows) > MAX_FILES:
        problems.append(f"파일 {len(rows)}개 > 최대 {MAX_FILES}개")
    if total > MAX_BYTES:
        problems.append(f"총 {total / 1024 ** 2:.1f}MB > 최대 {MAX_BYTES // 1024 ** 2}MB")
    if "code" not in kinds:
        problems.append("파이썬(.py) 파일이 없음")
    problems += [f"허용되지 않는 형식(.py·.csv·.txt·폰트만 가능): {n}" for n, k, _ in rows if k == "other"]
    problems += [f"파일명 규칙 위반(영문·숫자만): {n}" for n, _, _ in rows if not SAFE_NAME.match(n)]
    for f in files:
        if kind(f) == "code":
            try:
                ast.parse(f.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                problems.append(f"{f.name} 은(는) 올바른 파이썬 파일이 아님: {type(exc).__name__}")
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
