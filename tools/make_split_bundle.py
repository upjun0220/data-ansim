"""반입용 개별 코드 파일 만들기 — 센터 보안 점검(2026-10-08): 실행할 때 파일을 새로 만드는 묶음(ansimN.py)은 받지 않으니
코드 파일을 하나씩 따로 신청해야 한다.

같은 파일명은 다시 반입할 수 없다(2026-09-22 에 모듈을 원래 이름으로 반입한 적이 있다). 그래서 모듈 이름 끝에 버전 번호를 붙이고
(pipeline → pipeline10), 파일 안의 우리 모듈 import 문만 새 이름으로 바꾼다(import x → import x10 as x, from x import → from x10 import).
코드 내용은 그대로다. 진입점 fieldday1 → field{N}, 공휴일 달력 → holidays{N}.txt(내용은 config/holidays.yaml 과 같은 JSON).

사용: python tools/make_split_bundle.py <출력 폴더> [버전 번호, 기본 10]
센터(커널 재시작 후):
    import sys, os; sys.path.insert(0, os.path.dirname(r"field10.py의 Copy Path"))
    from field10 import run_first_visit, run_full
"""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_import_bundle import MODULES, ROOT  # noqa: E402


def names(v):
    """원래 모듈 이름 → 반입 이름(확장자 없음)."""
    out = {m: f"{m}{v}" for m in MODULES}
    out["fieldday1"] = f"field{v}"
    return out


def rename_imports(text, new):
    """우리 모듈 import 문만 새 이름으로 바꾼다. import x → import x10 as x (코드 안의 이름 x 는 그대로)."""
    alt = "|".join(sorted(map(re.escape, new), key=len, reverse=True))
    text = re.sub(rf"^(\s*)import ({alt})\b(\s+as\s+(\w+))?",
                  lambda m: f"{m[1]}import {new[m[2]]} as {m[4] or m[2]}", text, flags=re.M)
    return re.sub(rf"^(\s*)from ({alt}) import\b", lambda m: f"{m[1]}from {new[m[2]]} import", text, flags=re.M)


def build(out, v=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    new = names(v)
    files = {}
    for old, name in new.items():
        text = rename_imports((ROOT / f"{old}.py").read_text(encoding="utf-8").replace("\r\n", "\n"), new)
        ast.parse(text, filename=name)
        files[f"{name}.py"] = text
    files[f"holidays{v}.txt"] = (ROOT / "config" / "holidays.yaml").read_text(encoding="utf-8")
    for name, text in files.items():
        (out / name).write_bytes(text.encode("utf-8"))
    return out, new


def verify(out, new):
    """원래 이름 모듈이 없는 빈 폴더에서 진입점을 불러 본다(이름을 하나라도 덜 바꿨으면 ImportError)."""
    leftover = [f"{p.name}: {line.strip()}" for p in Path(out).glob("*.py") for line in p.read_text(encoding="utf-8").splitlines()
                if re.match(rf"\s*(import|from)\s+({'|'.join(map(re.escape, new))})\b(?!\d)", line)]
    if leftover:
        raise SystemExit("원래 이름 import 가 남음: " + "; ".join(leftover[:5]))
    with tempfile.TemporaryDirectory() as tmp:
        code = (f"import sys; sys.path.insert(0, {str(Path(out).resolve())!r}); "
                f"from {new['fieldday1']} import run_first_visit, run_full; print('불러오기 확인')")
        subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp, check=True)


if __name__ == "__main__":
    folder, mapping = build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    verify(folder, mapping)
    for p in sorted(folder.iterdir()):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
