"""현장 실행용 단일 코드 묶음(dayNbundle.py) 만들기 — 저장소 모듈 + fieldday1 + 공휴일 달력을 파일 하나에 담는다.

센터에서 `%run 파일이름.py` 하면 모듈을 day1code/ 에 풀고 run_first_visit·run_full 을 불러온다.
원본 데이터는 들어가지 않는다. 반입 파일명은 이전 반입과 겹치지 않게 번호를 올린다(day1bundle.py → 2 → 3 …).

사용: python tools/make_day1bundle.py [출력 .py]        (기본 dist/day1/day1bundle3.py)
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_import_bundle import MODULES, ROOT  # noqa: E402

EXTRA = {"fieldday1.py": ROOT / "fieldday1.py", "holidays.yaml": ROOT / "config" / "holidays.yaml"}

WRAPPER = '''target = Path.cwd() / "day1code"
target.mkdir(exist_ok=True)
if "fieldday1" in sys.modules:
    print("⚠ 이미 불러온 모듈이 있음 — Kernel → Restart 후 다시 실행해야 새 코드가 적용됨")
for name, source in FILES.items():
    (target / name).write_text(source, encoding="utf-8")
if str(target) not in sys.path:
    sys.path.insert(0, str(target))
from fieldday1 import run_first_visit, run_full
print("준비 완료: 다음 셀에서 run_first_visit(...) 또는 run_full(...)에 Copy Path를 넣으세요")
'''


def build(out):
    files = {f"{m}.py": (ROOT / f"{m}.py").read_text(encoding="utf-8") for m in MODULES}
    files.update({name: path.read_text(encoding="utf-8") for name, path in EXTRA.items()})
    files = {name: text.replace("\r\n", "\n") for name, text in files.items()}
    for name, text in files.items():
        if name.endswith(".py"):
            ast.parse(text, filename=name)
    text = ("# 안심데이터센터 현장 실행용 단일 코드 묶음 — tools/make_day1bundle.py 생성 파일, 직접 수정하지 말 것\n"
            "from pathlib import Path\nimport sys\n\n"
            f"FILES = {files!r}\n" + WRAPPER)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    return files


def verify(out):
    """빈 폴더에서 실제로 %run 처럼 실행해 모듈이 풀리고 불러와지는지 확인한다."""
    with tempfile.TemporaryDirectory() as tmp:
        code = f"import runpy; runpy.run_path({str(Path(out).resolve())!r}, run_name='bundle')"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp, check=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "day1" / "day1bundle3.py"
    files = build(out)
    verify(out)
    print(f"만듦: {out} · 파일 {len(files)}개 · SHA-256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
