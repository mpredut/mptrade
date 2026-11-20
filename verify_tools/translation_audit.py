#!/usr/bin/env python3
"""translation_audit.py -- helper for the English translation of comments and docstrings.

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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import parse_nonneg_cli_int, safe_nonneg_int
from state_io import atomic_write_text, read_text_file

# Detection dictionary. Diacritics are conclusive on their own; the word list holds
# Romanian tokens that are unlikely to appear in English prose or in identifiers.
_RO_DIACRITIC_SRC = "\u0103\u00e2\u00ee\u0219\u021b\u0102\u00c2\u00ce\u0218\u021a\u015f\u0163\u015e\u0162"
_RO_DIACRITIC_DST = "aaistAAISTstST"
RO_DIACRITICS = set(_RO_DIACRITIC_SRC)
_RO_DIACRITIC_MAP = dict(zip(_RO_DIACRITIC_SRC, _RO_DIACRITIC_DST))
RO_DIACRITIC_TRANS = str.maketrans(_RO_DIACRITIC_MAP)

_ASCII_TYPOGRAPHY_COMMON = {
    "\u2014": "--",
    "\u2013": "-",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2009": " ",
    "\u202f": " ",
    "\u200b": "",
    "\ufeff": "",
    "\u2192": "->",
    "\u2794": "->",
    "\u2190": "<-",
    "\u2194": "<->",
    "\u21d2": "=>",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2248": "~=",
    "\u00b1": "+/-",
    "\u00d7": "x",
    "\u00b7": "-",
    "\u2022": "-",
}
_ASCII_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}
_ASCII_PROSE_TRANS = str.maketrans({**_RO_DIACRITIC_MAP, **_ASCII_TYPOGRAPHY_COMMON, **_ASCII_QUOTES})
_ASCII_PROSE_NO_DIACRITIC_TRANS = str.maketrans({**_ASCII_TYPOGRAPHY_COMMON, **_ASCII_QUOTES})
_ASCII_DOCSTRING_TRANS = str.maketrans({**_RO_DIACRITIC_MAP, **_ASCII_TYPOGRAPHY_COMMON})
_ASCII_DOCSTRING_NO_DIACRITIC_TRANS = str.maketrans(_ASCII_TYPOGRAPHY_COMMON)
_FSTRING_START = getattr(tokenize, "FSTRING_START", -1)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", -1)
_FSTRING_END = getattr(tokenize, "FSTRING_END", -1)

TEXT_SUFFIXES = frozenset({
    ".conf", ".csv", ".env", ".example", ".html", ".ini", ".json", ".md", ".ps1",
    ".py", ".service", ".sh", ".toml", ".txt", ".yaml", ".yml",
})
TEXT_FILENAMES = frozenset({".gitattributes", ".gitignore", "submit"})
PROD_PYTHON_SKIP_TOP = frozenset({
    "tests", "offline", "verify_tools", "ops", ".venv", "venv", "myenv", ".git", "__pycache__",
})

DIACRITIC_EXEMPT_PATHS = frozenset({
    "orchestratorOS/admin/prune_llm_threads.py",
    "orchestratorOS/supervisor/prompt_formatters.py",
    "tests/test_autonomous_ai_reconciler.py",
    "tests/test_prune_llm_threads.py",
    "verify_tools/translation_audit.py",
})


def strip_romanian_diacritics(text: str) -> str:
    """Replace Romanian diacritics with their ASCII equivalents."""
    return text.translate(RO_DIACRITIC_TRANS)


def normalize_line_endings(text: str) -> str:
    """Normalize CRLF/CR line endings to Unix LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n") if "\r" in text else text


def normalize_ascii_prose(text: str, *, strip_diacritics: bool = True) -> str:
    """Replace Romanian diacritics and common Unicode punctuation/symbols with 7-bit ASCII."""
    return text.translate(_ASCII_PROSE_TRANS if strip_diacritics else _ASCII_PROSE_NO_DIACRITIC_TRANS)


def escape_non_ascii_literals(text: str) -> str:
    """Escape non-ASCII characters as Python \\uXXXX / \\UXXXXXXXX sequences."""
    if text.isascii():
        return text
    return "".join(
        ch if ord(ch) < 128 else (f"\\u{ord(ch):04x}" if ord(ch) <= 0xFFFF else f"\\U{ord(ch):08x}")
        for ch in text
    )


def is_diacritic_exempt(path: str) -> bool:
    """Return True when ``path`` is an intentional bilingual matcher/test file."""
    rel = os.path.relpath(os.path.abspath(path), _ROOT).replace(os.sep, "/")
    return rel in DIACRITIC_EXEMPT_PATHS or path.replace(os.sep, "/") in DIACRITIC_EXEMPT_PATHS


def is_text_hygiene_path(path: str) -> bool:
    """Return True when ``path`` is a tracked text file subject to LF hygiene."""
    base = os.path.basename(path)
    if base == ".env":
        return False
    if base in TEXT_FILENAMES:
        return True
    _, ext = os.path.splitext(base)
    return ext.lower() in TEXT_SUFFIXES


def is_prod_python_path(path: str, repo_root: str = _ROOT) -> bool:
    """Return True when ``path`` is a production Python module subject to 7-bit ASCII purity."""
    if not path.endswith(".py"):
        return False
    abs_root = os.path.abspath(repo_root)
    abs_path = os.path.abspath(path)
    try:
        in_repo = os.path.commonpath([abs_root, abs_path]) == abs_root
    except ValueError:
        in_repo = False
    if in_repo:
        rel_parts = os.path.relpath(abs_path, abs_root).replace(os.sep, "/").split("/")
        return not any(
            part in PROD_PYTHON_SKIP_TOP or part.startswith(".") or part == "site-packages"
            for part in rel_parts
        )
    parts = [p for p in abs_path.replace(os.sep, "/").split("/") if p]
    return not any(part in PROD_PYTHON_SKIP_TOP or part == "site-packages" for part in parts)


def _docstring_spans(source: str) -> set[Tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    has_runtime_doc = "__doc__" in source
    spans: set[Tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and getattr(node, "body", None):
            if isinstance(node, ast.Module) and has_runtime_doc:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                spans.add((first.value.lineno, first.value.col_offset))
    return spans


def sanitize_python_ascii(source: str, *, strip_diacritics: bool = True) -> str:
    """Normalize comments/docstrings to ASCII and escape non-ASCII in non-raw Python string literals."""
    if source.isascii():
        return source
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))
    doc_starts = _docstring_spans(source)
    doc_trans = _ASCII_DOCSTRING_TRANS if strip_diacritics else _ASCII_DOCSTRING_NO_DIACRITIC_TRANS
    fstring_raw_stack: List[bool] = []
    replacements: List[Tuple[int, int, str]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == _FSTRING_START:
                fstring_raw_stack.append("r" in tok.string.lower())
                continue
            if tok.type == _FSTRING_END:
                if fstring_raw_stack:
                    fstring_raw_stack.pop()
                continue
            if tok.type not in (tokenize.COMMENT, tokenize.STRING, _FSTRING_MIDDLE):
                continue
            start_idx = offsets[tok.start[0] - 1] + tok.start[1]
            end_idx = offsets[tok.end[0] - 1] + tok.end[1]
            raw_slice = source[start_idx:end_idx]
            if raw_slice.isascii():
                continue
            if tok.type == tokenize.COMMENT:
                new_slice = normalize_ascii_prose(raw_slice, strip_diacritics=strip_diacritics)
            elif tok.type == tokenize.STRING:
                q_pos = min((i for i in (raw_slice.find("'"), raw_slice.find('"')) if i >= 0), default=0)
                prefix = raw_slice[:q_pos].lower()
                if "r" in prefix or "b" in prefix:
                    continue
                if tok.start in doc_starts:
                    new_slice = escape_non_ascii_literals(raw_slice.translate(doc_trans))
                else:
                    new_slice = escape_non_ascii_literals(raw_slice)
            else:
                if not fstring_raw_stack or fstring_raw_stack[-1]:
                    continue
                new_slice = escape_non_ascii_literals(raw_slice)
            if new_slice != raw_slice:
                replacements.append((start_idx, end_idx, new_slice))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    if not replacements:
        return source
    out: List[str] = []
    cursor = 0
    for s, e, rep in replacements:
        out.append(source[cursor:s])
        out.append(rep)
        cursor = e
    out.append(source[cursor:])
    cleaned = "".join(out)
    try:
        ast.parse(cleaned)
    except SyntaxError:
        return source
    return cleaned


def find_first_non_ascii_line(text: str) -> int | None:
    """Return 1-based line number of the first non-ASCII line, or None if pure 7-bit ASCII."""
    if text.isascii():
        return None
    return next((idx for idx, ln in enumerate(text.splitlines(), 1) if not ln.isascii()), 1)


def sanitize_file_diacritics(path: str) -> bool:
    """Strip Romanian diacritics in-place for non-exempt text files; return True if modified."""
    if is_diacritic_exempt(path):
        return False
    source = read_text_file(path, default=None, newline="")
    if source is None:
        return False
    cleaned = strip_romanian_diacritics(source)
    if cleaned == source:
        return False
    atomic_write_text(path, cleaned, newline="", preserve_mode=True)
    return True


def sanitize_file_hygiene(path: str, *, repo_root: str = _ROOT) -> List[str]:
    """Auto-fix LF line endings, prod Python 7-bit ASCII, and Romanian diacritics in-place."""
    source = read_text_file(path, default=None, newline="")
    if source is None:
        return []
    cleaned = source
    actions: List[str] = []
    if "\r" in cleaned:
        cleaned = normalize_line_endings(cleaned)
        actions.append("CRLF->LF")
    if is_prod_python_path(path, repo_root=repo_root):
        next_cleaned = sanitize_python_ascii(cleaned, strip_diacritics=not is_diacritic_exempt(path))
        if next_cleaned != cleaned:
            cleaned = next_cleaned
            actions.append("ASCII")
    elif path.endswith(SCAN_EXTS) and not is_diacritic_exempt(path):
        next_cleaned = strip_romanian_diacritics(cleaned)
        if next_cleaned != cleaned:
            cleaned = next_cleaned
            actions.append("diacritics->ASCII")
    if cleaned != source:
        atomic_write_text(path, cleaned, newline="", preserve_mode=True)
    return actions


def ensure_git_hooks_configured(repo_root: str = _ROOT) -> bool:
    """Idempotently set ``core.hooksPath = .githooks`` when missing in ``.git/config``."""
    git_cfg = os.path.join(repo_root, ".git", "config")
    hooks_dir = os.path.join(repo_root, ".githooks")
    if not os.path.isfile(git_cfg) or not os.path.isdir(hooks_dir):
        return False
    cfg_text = read_text_file(git_cfg, default="")
    if "hooksPath = .githooks" in cfg_text:
        return True
    proc = subprocess.run(
        ["git", "-C", repo_root, "config", "core.hooksPath", ".githooks"],
        capture_output=True, check=False,
    )
    return proc.returncode == 0

# The list below was widened deliberately after the narrow version reported the
# migration finished three times while Romanian lines were still in the tree. It
# missed everything written without diacritics and with only one Romanian word per
# line (e.g. single-word status or price labels). Over-reporting costs a
# glance; under-reporting is what let those through.
RO_WORDS = frozenset("""
si sa se doar dar daca cand care este sunt nu pentru cu din sau mai foarte trebuie
adica deci astfel verifica verificat calculeaza calculat returneaza foloseste folosim
folosit avem aveam vrem putem poate trebuia facem face facut facuta facute
pretul pret preturi ordine ordin ordinul cumparare vanzare vinde vandut cumpara
cumparat pierdere pierderi castig castiguri prag praguri cadere soldul
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
decizie decizii rulare comision marime pozitie pozitii simulare oprita istoric
inchide inchis inchisa cicluri motorul combinatii difera intrarea umple
graficul graficele ferestrei glisante minimului maximului ajustam incrementam curatam plotam actualizam
moneda strategie valuta cumpararea cumparari plafon alocare pereche cheie cheia secrete
copiaza completeaza notificari bani reali forteaza levier lichidare sigur eticheta balanta
contractul auditului raporteaza ambele picioare picior orfan iesire inteligenta infrastructura
lacatul previne dublarea ordinelor succesul reseteaza contorul filtreaza ilichid negativ
claseaza corect limiteaza proaspat logica pozitia existenta ramane neatinsa plafonul
aditionali conversatiei functioneaza istoricul clar clara clare inapoi acolo cati ceea
totusi asadar potrivit datorita secunde saptamani urmatoarea anterioara adevarat
inactiva pornita deschis deschisa blocata permisa interzis interzisa ignorat ignorata
stearsa adaugata modificat modificata actualizat actualizata salvata incarcata trimisa
primita asteptat asteptata calculata verificata testat testata executata anulata
respins respinsa acceptat acceptata aprobat aprobata refuzata sintetic sintetica sintetice
""".split())

DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_IDENT_IGNORE = frozenset({"sub", "cand"})
_CAMEL_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
SKIP_DIRS = {".git", ".venv", "myenv", "__pycache__", "node_modules", ".pytest_cache", "cachedb", "logs", "data", "archive"}
SCAN_EXTS = (".py", ".sh", ".conf", ".env", ".example", ".md")
_TOKEN_RE = re.compile(r'`+[^`]+`+|"[^"\n]+"|\([a-zA-Z0-9_]+\)|[a-zA-Z_][a-zA-Z0-9_]*(?:[-/][a-zA-Z0-9_$]+)*')


# -- scan ----------------------------------------------------------------------

def romanian_score(text: str) -> int:
    """Heuristic score: 99 when diacritics are present, else the Romanian word count."""
    if any(ch in RO_DIACRITICS for ch in text):
        return 99
    words = _TOKEN_RE.findall(text.lower())
    return sum(1 for word in words if word in RO_WORDS)


_HASH_COMMENT_RE = re.compile(r"#.*$")


def scan_file(path: str, min_score: int) -> Tuple[List[tuple], bool]:
    """Return (hits, doc_is_runtime) for one file; a hit is (line, kind, score, text)."""
    source = read_text_file(path, default=None)
    if source is None:
        return [], False

    hits: List[tuple] = []
    if not path.endswith(".py"):
        for idx, line in enumerate(source.splitlines(), 1):
            text = line if path.endswith(".md") else (m.group(0) if (m := _HASH_COMMENT_RE.search(line)) else "")
            if text and (score := romanian_score(text)) >= min_score:
                hits.append((idx, "comment", score, text.strip()))
        return hits, False

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
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            id_words = [
                w.lower()
                for w in _CAMEL_WORD_RE.findall(node.name)
                if len(w) >= 3 and w.lower() not in _IDENT_IGNORE
            ]
            id_score = sum(1 for w in id_words if w in RO_WORDS)
            if id_score >= min_score:
                hits.append((getattr(node, "lineno", 0), "identifier", id_score, node.name))
        if isinstance(node, DOC_OWNERS):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            score = romanian_score(doc)
            if score >= min_score:
                hits.append((getattr(node, "lineno", 0), "docstring", score,
                             " ".join(doc.split())))

    return sorted(hits), "__doc__" in source


def iter_python_files(targets: Iterable[str]) -> Iterable[str]:
    for target in targets:
        if os.path.isfile(target):
            yield target
            continue
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name != ".env" and name.endswith(SCAN_EXTS):
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
                  " runtime CLI text -- do not translate it as ordinary prose.")
        shown = hits if args.all else hits[:safe_nonneg_int(args.limit, default=0)]
        for line, kind, score, text in shown:
            print(f"  L{line:<5} {kind:<9} score={score:<3} {text[:100]}")
        if len(hits) > len(shown):
            print(f"  ... {len(hits) - len(shown)} more (use --all)")

    print(f"\nTOTAL: {total_hits} block(s) across {total_files} scanned file(s)")
    return 1 if total_hits else 0


# -- verify --------------------------------------------------------------------

def code_fingerprint(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and getattr(node, "body", None):
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    return ast.dump(tree, include_attributes=False)


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
            work_source = read_text_file(path, newline="")
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
                      "  -- reopen with newline='' and redo the edit")

    print()
    print("RESULT:", "every file changed prose only" if not failures
          else f"{failures} file(s) changed more than prose")
    return 1 if failures else 0


# -- git hooks (pre-commit / commit-msg) ---------------------------------------

def _staged_hygiene_files(explicit_files: List[str] | None = None) -> List[str]:
    if explicit_files is not None:
        candidates = explicit_files
    else:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=_ROOT, capture_output=True, text=True, check=False,
        )
        candidates = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    selected: List[str] = []
    for rel in candidates:
        parts = set(rel.replace("\\", "/").split("/"))
        if parts & SKIP_DIRS or not is_text_hygiene_path(rel):
            continue
        full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
        if os.path.isfile(full):
            selected.append(full)
    return selected


def cmd_pre_commit(args) -> int:
    files = _staged_hygiene_files(args.files if args.files else None)
    total_hits = 0
    non_ascii_errors: List[str] = []
    for path in files:
        rel = os.path.relpath(path, _ROOT) if os.path.isabs(path) else path
        actions = sanitize_file_hygiene(path)
        if actions:
            print(f"[pre-commit] Auto-fixed ({', '.join(actions)}): {rel}")
            if not args.files:
                subprocess.run(["git", "add", "--", path], cwd=_ROOT, check=False)
        if is_prod_python_path(path):
            source = read_text_file(path, default=None, newline="")
            bad_ln = find_first_non_ascii_line(source) if source is not None else None
            if bad_ln is not None:
                non_ascii_errors.append(
                    f"{rel}:{bad_ln}: non-ASCII source character (use \\uXXXX/\\UXXXXXXXX escapes in runtime strings)"
                )
        if not path.endswith(SCAN_EXTS) or is_diacritic_exempt(path):
            continue
        hits = scan_file(path, args.min_score)[0]
        if not hits:
            continue
        total_hits += len(hits)
        print(f"\n=== {rel} ({len(hits)} Romanian block(s)) ===")
        for line, kind, score, text in hits:
            print(f"  L{line:<5} {kind:<9} score={score:<3} {text[:100]}")
    if non_ascii_errors:
        print("\n[pre-commit] Blocked: non-ASCII character(s) remain in production Python file(s):")
        for err in non_ascii_errors:
            print(f"  {err}")
    if total_hits:
        print(f"\n[pre-commit] Blocked: {total_hits} Romanian block(s) detected. "
              "Write comments, docstrings, and docs in English.")
    return 1 if (non_ascii_errors or total_hits) else 0


def cmd_commit_msg(args) -> int:
    source = read_text_file(args.msg_file, default=None, newline="")
    if source is None:
        return 0
    cleaned = normalize_ascii_prose(normalize_line_endings(source))
    if cleaned != source:
        atomic_write_text(args.msg_file, cleaned, newline="")
        print("[commit-msg] Auto-normalized commit message to 7-bit ASCII LF.")
    prose = "\n".join(
        line for line in cleaned.splitlines() if not line.lstrip().startswith("#")
    )
    if not prose.isascii():
        print("[commit-msg] Blocked: non-ASCII characters remain in commit message. "
              "Use 7-bit ASCII English.")
        return 1
    score = romanian_score(prose)
    if score >= args.min_score:
        matched = [w for w in _TOKEN_RE.findall(prose.lower()) if w in RO_WORDS]
        print(f"[commit-msg] Blocked: Romanian words detected in commit message "
              f"{matched} (score={score}). Write commit message in English.")
        return 1
    return 0


# -- entry point ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="report blocks that still look Romanian")
    scan.add_argument("targets", nargs="+", help="files or directories")
    scan.add_argument("--min-score", type=parse_nonneg_cli_int, default=2,
                      help="report a block at or above this score (default: 2)")
    scan.add_argument("--limit", type=parse_nonneg_cli_int, default=6,
                      help="blocks shown per file (default: 6)")
    scan.add_argument("--all", action="store_true", help="show every block")
    scan.set_defaults(func=cmd_scan)

    verify = sub.add_parser("verify", help="prove an edit touched only prose")
    verify.add_argument("paths", nargs="+", help="edited files")
    verify.add_argument("--rev", default="HEAD", help="revision to compare against")
    verify.set_defaults(func=cmd_verify)

    pre_commit = sub.add_parser(
        "pre-commit", help="auto-fix LF/ASCII/diacritics and verify staged files",
    )
    pre_commit.add_argument("files", nargs="*", help="optional explicit file paths")
    pre_commit.add_argument("--min-score", type=parse_nonneg_cli_int, default=2,
                            help="report a block at or above this score (default: 2)")
    pre_commit.set_defaults(func=cmd_pre_commit)

    commit_msg = sub.add_parser(
        "commit-msg", help="auto-normalize LF/ASCII/diacritics and verify commit message",
    )
    commit_msg.add_argument("msg_file", help="path to git commit message file")
    commit_msg.add_argument("--min-score", type=parse_nonneg_cli_int, default=2,
                            help="block commit message at or above this score (default: 2)")
    commit_msg.set_defaults(func=cmd_commit_msg)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
