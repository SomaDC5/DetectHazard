# -*- coding: utf-8 -*-
"""公開APIと引数を機械で抜き出す。**実装書の署名を出し直すための道具。**

なぜ要るか
----------
`docs/実装書.md` に署名を手で写してあるが、コードを直すと古くなる。
古くなったときに**全ファイルを読み直すのではなく、これで出し直す。**

    python tools/api_map.py                 # dc5lib と tools の公開API
    python tools/api_map.py --cli           # スクリプトの引数だけ
    python tools/api_map.py -o api.txt      # ファイルに書く（既定は標準出力）

出すのは**公開のもの（`_` で始まらない）と大文字の定数**だけである。
中身は出さない ── 実装書に必要なのは「何を呼べばよいか」までで、
**どう書いてあるかは読まない**のが方針（`docs/実装書.md` の冒頭）。
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NL = chr(10)
out = io.StringIO()


def sig(fn):
    """def の引数を `name(a, b=1, *c, **d)` の形にする。"""
    a = fn.args
    parts = []
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, dv in zip(a.args, defaults):
        if dv is None:
            parts.append(arg.arg)
        else:
            try:
                parts.append(f"{arg.arg}={ast.unparse(dv)}")
            except Exception:                           # noqa: BLE001
                parts.append(f"{arg.arg}=...")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    if a.kwonlyargs and not a.vararg:
        parts.append("*")
    for arg, dv in zip(a.kwonlyargs, a.kw_defaults):
        try:
            parts.append(f"{arg.arg}=" + (ast.unparse(dv) if dv else ""))
        except Exception:                               # noqa: BLE001
            parts.append(arg.arg + "=...")
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def first_line(node):
    for ln in (ast.get_docstring(node) or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def consts(tree):
    """大文字の定数。**値が長いときは切る**（中身は読まない方針）。"""
    rows = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    try:
                        v = ast.unparse(n.value)
                    except Exception:                   # noqa: BLE001
                        v = "..."
                    rows.append((t.id, v[:90]))
    return rows


def dump(label, files):
    out.write(f"{'=' * 78}{NL}{label}{NL}{'=' * 78}{NL}")
    for p in files:
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            out.write(f"{NL}--- {p.name}  **構文エラー {e}**{NL}")
            continue
        out.write(f"{NL}--- {p.relative_to(ROOT).as_posix()}  "
                  f"{len(src.splitlines())}行{NL}")
        head = first_line(tree)
        if head:
            out.write(f"    {head}{NL}")
        for name, val in consts(tree):
            out.write(f"    [定数] {name} = {val}{NL}")
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                dec = [ast.unparse(d) for d in n.decorator_list]
                tag = (" @" + ",".join(dec)) if dec else ""
                out.write(f"    class {n.name}{tag}  {first_line(n)}{NL}")
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if m.name.startswith("_") and m.name != "__init__":
                            continue
                        pr = "@property " if any(
                            ast.unparse(d) == "property"
                            for d in m.decorator_list) else ""
                        out.write(f"        {pr}.{sig(m)}"
                                  f"  {first_line(m)}{NL}")
                    elif isinstance(m, ast.AnnAssign) and isinstance(
                            m.target, ast.Name):
                        try:
                            out.write(f"        .{m.target.id}: "
                                      f"{ast.unparse(m.annotation)}{NL}")
                        except Exception:               # noqa: BLE001
                            pass
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name.startswith("_"):
                    continue
                out.write(f"    def {sig(n)}{NL}")
                fl = first_line(n)
                if fl:
                    out.write(f"        {fl}{NL}")


def dump_cli(label, files):
    """argparse の引数だけを並べる。スクリプトの入口を一覧で見るため。"""
    out.write(f"{'=' * 78}{NL}{label}  — スクリプトの引数{NL}{'=' * 78}{NL}")
    for p in files:
        src = p.read_text(encoding="utf-8")
        if "add_argument" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        args = sorted(set(re.findall(r"""add_argument\(\s*["'](--?[\w-]+)""",
                                     src)))
        out.write(f"{NL}{p.relative_to(ROOT).as_posix()}  "
                  f"{first_line(tree)[:70]}{NL}")
        out.write(f"    {' '.join(args)}{NL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true",
                    help="argparse の引数だけを出す")
    ap.add_argument("-o", "--out", default=None,
                    help="書き出すファイル。省略すると標準出力")
    a = ap.parse_args()

    if a.cli:
        for d in ("analysis", "tools"):
            dump_cli(d, sorted((ROOT / d).glob("*.py")))
    else:
        dump("dc5lib", sorted((ROOT / "dc5lib").glob("*.py")))
        dump("tools", sorted((ROOT / "tools").glob("*.py")))

    txt = out.getvalue()
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"{a.out} に {len(txt.splitlines())} 行 / {len(txt):,} 文字")
    else:
        sys.stdout.write(txt)


if __name__ == "__main__":
    main()
