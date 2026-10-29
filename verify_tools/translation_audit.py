#!/usr/bin/env python3
"""translation_audit.py — helper for the English translation of comments and docstrings.

Two steps: find what still has to be translated, then prove an edit touched nothing but
prose. The bulk migration is finished; this stays as a guard against regressions.

Subcommands
  scan    Report comment and docstring blocks that still look Romanian, and flag
          module docstrings that are reused as runtime CLI text (argparse), which
          must not be translated as if they were ordinary prose.
  verify  Prove an edit is prose-only: compare the worktree file against its HEAD
          version after stripping true docstrings from both ASTs, and confirm the
          line terminators were preserved.

Both subcommands exit non-zero when they find something, so they compose with CI.

Examples
  python3 verify_tools/translation_audit.py scan verify_tools/
  python3 verify_tools/translation_audit.py scan tests/ --min-score 3
  python3 verify_tools/translation_audit.py verify log.py instruments_config.py
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from typing import Iterable, List, Tuple

# Detection dictionary. Diacritics are conclusive on their own; the word list holds
# Romanian tokens that are unlikely to appear in English prose or in identifiers.
RO_DIACRITICS = set("ăâîșțĂÂÎȘȚşţŞŢ")

# The list below was widened deliberately after the narrow version reported the
# migration finished three times while Romanian lines were still in the tree. It
# missed everything written without diacritics and with only one Romanian word per
# line: "esuat", "lipsa", "astept", "pretul", "coada plina". Over-reporting costs a
# glance; under-reporting is what let those through.
RO_WORDS = frozenset("""
si sa se doar dar daca cand care este sunt nu pentru cu din sau mai foarte trebuie
adica deci astfel verifica verificat calculeaza calculat returneaza foloseste folosim
folosit avem aveam vrem putem poate trebuia facem face facut facuta facute
pretul pret preturi ordine ordin ordinul cumparare vanzare vinde vandut cumpara
cumparat pierdere pierderi castig castiguri prag praguri cadere sold soldul
fisier fisierul fisiere fereastra ferestre valoare valori cantitate cantitati
toate toti toata fiecare acelasi aceeasi acest aceasta aceste acesti niciun nicio
inainte dupa acum mereu niciodata deja inca apoi atunci astfel
nou noua noi vechi veche prima primul ultima ultimul intre peste sub langa
scrie scris citeste citit sterge sters adauga adaugat porneste pornit opreste oprit
reporneste repornit incarca incarcat salveaza salvat trimite trimis primeste primit
ruleaza rulat plaseaza plasat executa executat anuleaza anulat blocheaza blocat
permite permis refuza refuzat esuat esuata esec reusit gasit gasita lipsa lipseste
astept asteapta asteptare gol goala plina mort moarte viu curat coada
fara catre dintre despre prin asupra impotriva orice oricare nimic nimeni
motiv motive cauza scop rand randuri linie linii pas pasi etapa etape exemplu exemple
""".split())

DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
SKIP_DIRS = {".git", ".venv", "myenv", "__pycache__", "node_modules", ".pytest_cache"}


# ── scan ──────────────────────────────────────────────────────────────────────

def romanian_score(text: str) -> int:
    """Heuristic score: 99 when diacritics are present, else the Romanian word count."""
    if any(ch in RO_DIACRITICS for ch in text):
        return 99
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for word in words if word in RO_WORDS)


def doc_is_runtime_text(source: str) -> bool:
    """True when the module docstring is reused as runtime text, e.g. passed to argparse.

    Translating such a docstring changes user-visible CLI output, so it is not a
    prose-only edit. tradeall_price_archiver.py is the known case in this repository.
    """
    return "__doc__" in source


def scan_file(path: str, min_score: int) -> Tuple[List[tuple], bool]:
    """Return (hits, doc_is_runtime) for one file; a hit is (line, kind, score, text)."""
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except (OSError, UnicodeDecodeError):
        return [], False

    hits: List[tuple] = []

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                score = romanian_score(token.string)
                if score >= min_score:
                    hits.append((token.start[0], "comment", score,
                                 token.string.strip()))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sorted(hits), False

    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            score = romanian_score(doc)
            if score >= min_score:
                hits.append((getattr(node, "lineno", 0), "docstring", score,
                             " ".join(doc.split())))

    return sorted(hits), doc_is_runtime_text(source)


def iter_python_files(targets: Iterable[str]) -> Iterable[str]:
    for target in targets:
        if os.path.isfile(target):
            yield target
            continue
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(root, name)


def cmd_scan(args) -> int:
    total_hits = 0
    total_files = 0
    for path in iter_python_files(args.targets):
        total_files += 1
        hits, runtime_doc = scan_file(path, args.min_score)
        if not hits:
            continue
        total_hits += len(hits)
        print(f"\n=== {path}  ({len(hits)} block(s)) ===")
        if runtime_doc:
            print("    WARNING: this file references __doc__; its module docstring may be"
                  " runtime CLI text — do not translate it as ordinary prose.")
        shown = hits if args.all else hits[:max(0, int(args.limit))]
        for line, kind, score, text in shown:
            print(f"  L{line:<5} {kind:<9} score={score:<3} {text[:100]}")
        if len(hits) > len(shown):
            print(f"  ... {len(hits) - len(shown)} more (use --all)")

    print(f"\nTOTAL: {total_hits} block(s) across {total_files} scanned file(s)")
    return 1 if total_hits else 0


# ── verify ────────────────────────────────────────────────────────────────────

def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove true docstrings, so only executable code remains for comparison."""
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and getattr(node, "body", None):
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    return tree


def code_fingerprint(source: str) -> str:
    return ast.dump(strip_docstrings(ast.parse(source)), include_attributes=False)


def line_ending_style(source: str) -> str:
    """Classify line terminators as "crlf", "lf", "none" or "mixed(...)".

    The style is compared rather than raw counts, because a prose-only edit may
    legitimately add or remove lines, for example when a docstring is rewrapped.
    """
    crlf = source.count("\r\n")
    lf_only = source.count("\n") - crlf
    if crlf and not lf_only:
        return "crlf"
    if lf_only and not crlf:
        return "lf"
    if not crlf and not lf_only:
        return "none"
    return f"mixed({crlf} crlf / {lf_only} lf)"


def cmd_verify(args) -> int:
    failures = 0
    for path in args.paths:
        try:
            # Bytes, not text: subprocess text mode translates CRLF to LF and would
            # hide exactly the whole-file rewrite this check exists to catch.
            head_source = subprocess.run(
                ["git", "show", f"{args.rev}:{path}"],
                capture_output=True, check=True).stdout.decode("utf-8")
        except subprocess.CalledProcessError:
            print(f"FAIL {path:<40} not found in {args.rev}")
            failures += 1
            continue

        try:
            with open(path, encoding="utf-8", newline="") as handle:
                work_source = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"FAIL {path:<40} unreadable on disk ({exc})")
            failures += 1
            continue

        try:
            same_code = code_fingerprint(head_source) == code_fingerprint(work_source)
        except SyntaxError:
            same_code = False
        head_endings = line_ending_style(head_source)
        work_endings = line_ending_style(work_source)
        same_endings = head_endings == work_endings

        if same_code and same_endings:
            print(f"OK   {path:<40} prose-only edit")
        else:
            failures += 1
            print(f"FAIL {path:<40} code_identical={same_code} "
                  f"line_endings_identical={same_endings}")
            if not same_endings:
                print(f"       HEAD={head_endings}  worktree={work_endings}"
                      "  — reopen with newline='' and redo the edit")

    print()
    print("RESULT:", "every file changed prose only" if not failures
          else f"{failures} file(s) changed more than prose")
    return 1 if failures else 0


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="report blocks that still look Romanian")
    scan.add_argument("targets", nargs="+", help="files or directories")
    scan.add_argument("--min-score", type=int, default=2,
                      help="report a block at or above this score (default: 2)")
    scan.add_argument("--limit", type=int, default=6,
                      help="blocks shown per file (default: 6)")
    scan.add_argument("--all", action="store_true", help="show every block")
    scan.set_defaults(func=cmd_scan)

    verify = sub.add_parser("verify", help="prove an edit touched only prose")
    verify.add_argument("paths", nargs="+", help="edited files")
    verify.add_argument("--rev", default="HEAD", help="revision to compare against")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
