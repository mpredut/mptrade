"""Specialized Multi-Source Feeds & Dynamic Crisis Lexicon Discovery (SSOT sub-module)."""
from __future__ import annotations

import logging
import os
import re
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from botcore import (
    normalize_epoch_sec,
    obj_attr,
    resolve_now_ts,
    safe_float,
    split_csv as _split_csv,
)
from cachepaths import cache_path, resolve_cachedb_filepath
from state_io import read_json_dict_cached, save_json_state

_ssot: Any = None

logger = logging.getLogger(__name__)

_LEXICON_FILE_CACHE: Dict[str, Tuple[tuple, Dict[str, Any]]] = {}
_LEXICON_EVAL_CACHE: Dict[Tuple[Any, ...], Dict[str, Tuple[str, ...]]] = {}

_NOISE_LEXICON_PHRASES: frozenset[str] = frozenset({
    "more than", "first time", "yahoo finance", "financial times", "new york",
    "york times", "wall street", "street journal", "jerusalem post", "military times",
    "associated press", "was heartbroken", "leave everything", "united states",
    "middle east", "according pentagon", "pentagon estimate", "cuba wins",
    "shocks however", "profit thanks", "thanks gold", "insane plan", "focus live",
    "doordash strikes", "billion deal", "rival instacart", "grocery delivery",
    "gillibrand presses", "civilian schools", "hacked emergency", "shock emergency",
    "emergency sanction", "supply shock", "warfare cyber",
})

_NON_CRISIS_STRIKE_CONTEXT_WORDS: frozenset[str] = frozenset({
    "deal", "deals", "agreement", "partnership", "pact", "contract", "bargain",
    "bet", "chord", "balance", "gold", "rich", "pose", "lightning",
    "protest", "protests", "union", "unions", "workers", "rail", "transit",
})

_NON_CRISIS_ATTACK_CONTEXT_WORDS: frozenset[str] = frozenset({
    "allegedly", "alligator", "animal", "arrest", "arrested", "baseball", "bat",
    "bear", "boy", "burglary", "charged", "cougar", "dog", "grizzly", "heart",
    "lion", "mugger", "panic", "robbery", "shark", "stabbing", "suspect",
    "teenager", "tiger", "volunteer", "wildlife", "wolf",
})

_PARTISAN_ELECTION_CONTEXT_WORDS: frozenset[str] = frozenset({
    "campaign", "congressional", "democrat", "democrats", "desantis", "election",
    "elections", "gop", "midterm", "midterms", "poll", "polling", "polls",
    "primary", "republican", "republicans", "vance", "voter", "voters",
})

_HUMAN_INTEREST_WAR_FILLER_WORDS: frozenset[str] = frozenset({
    "alaska", "alaskan", "alaskans", "bills", "cabin", "cabins", "cambodia",
    "centcom", "cruise", "cruises", "dreading", "forever", "french", "fries",
    "heating", "passengers", "podcast", "podcasts", "residents", "retirees",
    "retirement", "social", "tourist", "tourists", "vacation",
})

_COMMERCIAL_METAPHOR_WAR_WORDS: frozenset[str] = frozenset({
    "bidding", "browser", "burger", "burgers", "console", "coupon", "coupons",
    "culture", "discount", "discounts", "food", "mcdonald", "mcdonalds", "menu",
    "menus", "streaming", "talent", "turf",
})

_SPORTS_CONTEXT_WORDS: frozenset[str] = frozenset({
    "basketball", "championship", "coach", "cricket", "fifa", "football",
    "goalkeeper", "golf", "hockey", "laliga", "liv", "mlb", "nba", "nfl",
    "pga", "playoff", "playoffs", "premier", "rugby", "soccer", "stadium",
    "striker", "tenerife", "tennis", "tournament", "uefa",
})

_KINETIC_OR_MACRO_WAR_CONTEXT_WORDS: frozenset[str] = frozenset({
    "airstrike", "airstrikes", "attack", "attacks", "blockade", "bomb", "bombing",
    "casualties", "ceasefire", "crude", "drone", "drones", "embargo", "escalation",
    "hormuz", "idf", "invasion", "missile", "missiles", "military", "nuclear",
    "oil", "pentagon", "retaliation", "sanction", "sanctions", "strike", "strikes",
    "tanker", "tehran", "troops", "warhead", "warship",
})

_SINGLE_STOCK_EARNINGS_CONTEXT_WORDS: frozenset[str] = frozenset({
    "apparel", "earnings", "eps", "guidance", "merchandise", "pinterest",
    "quarterly", "retailer", "revenue", "shopify", "subscriber", "subscribers",
})

_BROAD_MARKET_INDEX_CONTEXT_WORDS: frozenset[str] = frozenset({
    "bitcoin", "crypto", "dax", "dow", "equities", "ftse", "futures", "global",
    "index", "indices", "market", "markets", "nasdaq", "nikkei", "sp500", "stocks",
    "treasury", "wall",
})

_TRACKING_QUERY_PARAMS: frozenset[str] = frozenset({
    "at_campaign", "at_medium", "mod", "oc", "rss", "traffic_source",
    "utm_campaign", "utm_content", "utm_medium", "utm_source", "utm_term",
})


def canonical_article_url(url: Any) -> str:
    """Normalize an article URL by stripping RSS tracking query parameters and trailing slashes."""
    if not isinstance(url, str) or not url.strip():
        return ""
    raw = url.strip()
    try:
        parsed = urllib.parse.urlsplit(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw.rstrip("/")
        kept_q = [
            (k, v)
            for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_QUERY_PARAMS
        ]
        clean_path = parsed.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            urllib.parse.urlencode(kept_q),
            "",
        ))
    except Exception:
        return raw.rstrip("/")


_STEM_EXACT_PREFIX_WHITELIST: Dict[str, frozenset[str]] = {
    "ban": frozenset({"ban", "bans", "banned", "banning"}),
    "freeze": frozenset({"freeze", "freezes", "freezing", "frozen"}),
    "war": frozenset({
        "war", "wars", "warfare", "wartime", "warship", "warships",
        "warhead", "warheads", "warplane", "warplanes", "warzone", "warring",
    }),
    "hack": frozenset({"hack", "hacks", "hacked", "hacker", "hackers", "hacking"}),
    "halt": frozenset({"halt", "halts", "halted", "halting"}),
    "strike": frozenset({"strike", "strikes", "striking"}),
}


_SEED_SETS_CACHE: Dict[str, Tuple[Any, Set[str]]] = {}


def _get_stopwords_set() -> Set[str]:
    raw = _ssot._cfg_str("lexicon_seed", "stopwords")
    cached = _SEED_SETS_CACHE.get("stopwords")
    if cached is not None and cached[0] == raw:
        return cached[1]
    res = set(_split_csv(raw, lower=True))
    _SEED_SETS_CACHE["stopwords"] = (raw, res)
    return res


def _get_crisis_stems_set() -> Set[str]:
    raw = _ssot._cfg_str("lexicon_seed", "crisis_context_stems")
    cached = _SEED_SETS_CACHE.get("crisis_stems")
    if cached is not None and cached[0] == raw:
        return cached[1]
    res = set(_split_csv(raw, lower=True))
    _SEED_SETS_CACHE["crisis_stems"] = (raw, res)
    return res


def _get_seed_terms_set() -> Set[str]:
    raw_h = _ssot._cfg_str("lexicon_seed", "high_severity_terms")
    raw_a = _ssot._cfg_str("lexicon_seed", "acute_shock_phrases")
    key_sig = (raw_h, raw_a)
    cached = _SEED_SETS_CACHE.get("seed_terms")
    if cached is not None and cached[0] == key_sig:
        return cached[1]
    res = set(_split_csv(raw_h, lower=True)) | set(_split_csv(raw_a, lower=True))
    _SEED_SETS_CACHE["seed_terms"] = (key_sig, res)
    return res


def _token_matches_crisis_stem(tok: str, crisis_stems: Set[str]) -> bool:
    """Return True if ``tok`` is a genuine derivative of a configured crisis stem (avoiding substring false positives)."""
    w = (tok or "").strip().lower()
    if len(w) < 3:
        return False
    return any(
        (w in allowed if (allowed := _STEM_EXACT_PREFIX_WHITELIST.get(stem)) is not None else w.startswith(stem))
        for stem in crisis_stems
    )


_NON_CRISIS_DYNAMIC_BIGRAM_WORDS: frozenset[str] = frozenset({
    "acknowledge", "acknowledges", "active", "activity", "adds", "admit", "admits",
    "adverse", "affecting", "ahead", "airfares", "alerts", "allegedly",
    "already", "amid", "answered", "app", "apparently", "appears", "apps",
    "arrested", "article", "articles", "baseball", "bat", "belgium", "below",
    "blame", "blamed", "blames", "blaming", "bondi", "boy", "brake",
    "breaking", "briefing", "bus", "calls", "car", "carried", "carries",
    "cascade", "caused", "ceasefire", "chip", "commentary", "concede",
    "concedes", "conduct", "conducted", "conducts", "continue", "continued",
    "continues", "contagion", "criticize", "criticized", "criticizes",
    "damaged", "deadly", "debunked", "deepening", "defensive", "digested", "discourse",
    "discount", "discounts", "downscale", "efforts", "eight", "elevated",
    "end", "essay", "essays", "eve", "exceptionally", "exists", "expert",
    "experts", "expiration", "explained", "exposes", "faces", "facing",
    "fallout", "false", "fare", "features", "five", "fleet", "florida",
    "forced", "four", "fresh", "functioning", "golf", "headlines", "heighten",
    "highlights", "history", "horrific", "humanitarian", "hurt", "intel",
    "institute", "introduces", "involvement", "island", "jet", "join",
    "keyword", "kinetic", "landing", "latest", "likely", "limited", "lingering", "links",
    "liquidation", "liquidations", "liv", "localized", "long", "loser",
    "losers", "losses", "macro", "media", "members", "midterm", "midterms",
    "mobile", "moderate", "mounting", "mutual", "narrative", "nine", "non", "normal",
    "noting", "official", "officials", "one", "orderly", "outlook",
    "peacefully", "persistent", "physical", "plane", "playbook", "played", "plunge",
    "plunged", "plunges", "plunging", "political", "position", "post",
    "potential", "prices", "processed", "progress", "prolongation", "pursues",
    "qeshm", "questions", "rather", "readiness", "record", "reflect",
    "reflects", "related", "renewed", "reported", "reportedly", "represent",
    "researchers", "reshapes", "retaining", "rhetoric", "risk", "risks",
    "role", "rout", "routine", "ryanair", "say", "scrutiny", "secondary",
    "selloff", "seven", "short", "showdown", "six", "sizing", "soaring", "somalia",
    "south", "stocks", "story", "strategy", "striker", "study", "surging", "suspected",
    "sustain", "sustained", "sustaining", "sustains", "sweden", "systemic", "tactical", "targets",
    "tech", "techspot", "ten", "tenerife", "three", "timeline", "trace",
    "traced", "train", "trains", "truck", "trump", "two", "uganda",
    "unabsorbed", "unanticipated", "uncertainty", "unpriced", "veto",
    "victim", "victims", "volatility", "volunteer", "wave", "waves", "widespread",
    "withdrawing", "workers", "worse", "wrap",
})


def _extract_contiguous_token_runs(text: str, stopwords: Set[str]) -> List[List[str]]:
    """Split ``text`` into contiguous runs of >=3-char non-stopword/non-noise alphabetic tokens."""
    runs: List[List[str]] = []
    cur: List[str] = []
    for raw_tok in re.findall(r"[a-z]+|[^\s\w]+|\d+", (text or "").lower()):
        if raw_tok.isalpha() and len(raw_tok) >= 3 and raw_tok not in stopwords and raw_tok not in _NON_CRISIS_DYNAMIC_BIGRAM_WORDS:
            cur.append(raw_tok)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _is_noise_dynamic_keyword(term: Any, stopwords: Optional[Set[str]] = None) -> bool:
    """Return True if ``term`` is a generic stopword/publisher/lifestyle phrase that must not trigger [HIGH-SEV]."""
    if not isinstance(term, str):
        return True
    t = re.sub(r"\s+", " ", term.strip().lower())
    sw = stopwords if stopwords is not None else _get_stopwords_set()
    if len(t) < 4 or t in _NOISE_LEXICON_PHRASES or t in sw:
        return True
    words = re.findall(r"[a-zA-Z]{2,}", t)
    return not words or any(w in sw or w in _NON_CRISIS_DYNAMIC_BIGRAM_WORDS for w in words)


def _dynamic_keyword_has_crisis_stem(term: str, crisis_stems: Optional[Set[str]] = None) -> bool:
    """Return True if at least one token in ``term`` matches a configured crisis stem."""
    stems = crisis_stems if crisis_stems is not None else _get_crisis_stems_set()
    return any(_token_matches_crisis_stem(w, stems) for w in re.findall(r"[a-zA-Z]{3,}", (term or "").lower()))


def _is_excluded_non_crisis_attack_keyword(
    kw_clean: str,
    text_words: Set[str],
    crisis_stems: Set[str],
) -> bool:
    """Return True if a keyword appears solely in a non-crisis local crime, partisan election, commercial metaphor, sports, or single-stock earnings context."""
    kw_words = set(re.findall(r"[a-zA-Z]{3,}", (kw_clean or "").lower()))
    combined_words = text_words | kw_words
    kinetic_only = _KINETIC_OR_MACRO_WAR_CONTEXT_WORDS - {
        "crude", "oil", "ceasefire", "escalation", "tehran", "pentagon", "idf", "military", "nuclear",
    }
    if (combined_words & _COMMERCIAL_METAPHOR_WAR_WORDS) and not (text_words & (kinetic_only - kw_words)):
        if kw_words and (kw_words & {"war", "wars"}):
            return True
    if (combined_words & _SPORTS_CONTEXT_WORDS) and not (text_words & (kinetic_only - kw_words)):
        return True
    if (combined_words & _NON_CRISIS_STRIKE_CONTEXT_WORDS) and kw_words and kw_words <= {"strike", "strikes"} and not (text_words & (kinetic_only - {"strike", "strikes"})):
        return True
    if kw_words and kw_words <= {
        "war", "wars", "conflict", "escalation", "iran", "israel", "ukraine",
        "russia", "gaza", "middle", "east", "energy", "crisis", "nuclear",
    }:
        if (
            combined_words
            & (
                _PARTISAN_ELECTION_CONTEXT_WORDS
                | _HUMAN_INTEREST_WAR_FILLER_WORDS
                | {
                    "strategy", "playbook", "outlook", "parse", "parsing",
                    "before", "already", "scrutiny", "rumors", "prolongation",
                    "electric", "vehicle", "vehicles",
                }
            )
        ) and not (combined_words & ((kinetic_only - kw_words) | _BROAD_MARKET_INDEX_CONTEXT_WORDS)):
            return True
        if (
            combined_words
            & {
                "diplomatic", "diplomacy", "peace", "resolution", "truce",
                "halt", "halts", "halted", "halting", "deescalation", "deescalate",
                "avoid", "avoids", "ease", "eases", "easing", "end", "ends", "ended", "ending",
                "talks", "negotiation", "negotiations", "proposal", "proposals", "reviewing",
            }
        ) and not (
            combined_words
            & (
                (kinetic_only - kw_words)
                | {
                    "collapse", "collapses", "collapsed", "fail", "fails", "failed",
                    "reject", "rejects", "rejected", "violate", "violates", "violated",
                    "resume", "resumes", "resumed", "abandon", "abandons", "abandoned",
                    "shatter", "shatters", "shattered", "threat", "threatens", "threatened",
                }
            )
        ):
            return True
    if kw_words and kw_words <= {"default", "risk", "risks", "insolvency"}:
        if (
            combined_words
            & (
                _PARTISAN_ELECTION_CONTEXT_WORDS
                | _SINGLE_STOCK_EARNINGS_CONTEXT_WORDS
                | {
                    "democratic", "sweep", "rout", "spacex", "spcx", "musk", "elon",
                    "tesla", "nvda", "nvidia", "rgnt", "regentis", "student", "mortgage",
                }
            )
        ) and not (combined_words & {"sovereign", "treasury", "government", "ceiling", "bank", "banks", "exchange"}):
            return True
    if kw_words and kw_words <= {"binance", "crypto", "exchange", "hack", "hacks", "hacked", "hacker", "hackers"}:
        if (
            combined_words
            & {"denies", "denied", "probe", "fbi", "indicted", "sentenced", "trial", "provocation", "trite"}
        ) and not (
            combined_words & {"drained", "stolen", "exploited", "halted", "suspended", "withdrawals", "breach"}
        ):
            return True
    if "stock" in kw_words and any(w.startswith("plung") for w in kw_words):
        if (combined_words & _SINGLE_STOCK_EARNINGS_CONTEXT_WORDS) and not (combined_words & _BROAD_MARKET_INDEX_CONTEXT_WORDS):
            return True
    if not (combined_words & _NON_CRISIS_ATTACK_CONTEXT_WORDS):
        return False
    other_stems = crisis_stems - {"attack", "kill"}
    if any(_token_matches_crisis_stem(w, other_stems) for w in kw_words):
        return False
    if text_words and any(_token_matches_crisis_stem(w, other_stems) for w in text_words):
        return False
    return any(w.startswith("attack") for w in kw_words)


_EDITORIAL_CLICKBAIT_SUBSTRINGS: Tuple[str, ...] = (
    "stocks to buy",
    "stocks to watch",
    "best stocks to",
    "top stocks to",
    "value trap",
    "is it a bargain",
    "running out of gas",
    "biggest stock losers",
    "biggest stock winners",
    "good bets in",
    "best bets for",
)

_EDITORIAL_CLICKBAIT_OR_ADVICE_RE = re.compile(
    r"^(?:['‘\"].*['’\"]:\s*)?(?:i(?:['’]m|['’]ve|\s+have|\s+am)\s+(?:\d{2}\b|worth\b|retired\b|inheriting\b|saving\b)|we(?:['’]re|['’]ve|\s+have|\s+are)\s+(?:\d{2}\b|in\s+our\b|worth\b|retired\b)|my\s+(?:(?:[\d-]+-year-old|elderly|late|aging)\s+)?(?:husband|wife|spouse|partner|parents|mother|father|brother|sister|son|daughter|adviser|advisor|financial\s+adviser|financial\s+advisor))\b|^['‘\"].*['’\"]:\s*(?:i(?:['’]m|['’]ve|\s+have|\s+am)|we(?:['’]re|['’]ve|\s+have|\s+are))\b|^cramer:\s|^prediction:\s|^best\s+[a-z0-9-]+\s+dip\s+buys\b|^not\s+[a-z]+\.\s+not\s+[a-z]+\.|^historically\s+(?:bullish|bearish)\s+signal\s+emerges\s+for\b|\(hint:\s*only\s+\d+\s+is\s+a\s+(?:buy|sell)\)|\bis\s+it\s+(?:time\s+to\s+buy|a\s+bargain\s+or\s+a\s+value\s+trap)\?|\b(?:can i buy a\s+\$?\d|can we afford\s+a\s+\$?\d|how to buy a\s+(?:new\s+|used\s+)?(?:car|home|house)\b|should i stop going|pushing roth conversions|we.re in our|annuity sales pitch|what politician speaks for me|worst-performing s&p 500 stock is finally seeing|a massive trade just happened in a leading|most investors are blind to|here is the [a-z0-9-]+(?:\s+strategy|\s+plan|\s+way)\s+to\s+survive|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:[a-z0-9&-]+\s+){0,3}stocks?\s+to\s+(?:buy|sell|watch|avoid))\b"
)


def _is_editorial_clickbait_or_advice_title(title_low: str) -> bool:
    """Return True if a lowercase headline matches first-person financial advice or stock-pick clickbait."""
    if not title_low:
        return False
    return any(p in title_low for p in _EDITORIAL_CLICKBAIT_SUBSTRINGS) or bool(
        _EDITORIAL_CLICKBAIT_OR_ADVICE_RE.search(title_low)
    )


def filter_valid_severity_keywords(
    raw_keywords: Any,
    *,
    title: Optional[str] = None,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Filter a headline's matched keywords, stripping generic stopword/publisher noise while keeping seed terms."""
    seed_set = _get_seed_terms_set()
    dyn_acute_set = set(load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts).get("acute_shock_phrases", ()))
    crisis_stems = _get_crisis_stems_set()
    sw = _get_stopwords_set()
    title_low = title.lower() if isinstance(title, str) and title.strip() else ""
    if _is_editorial_clickbait_or_advice_title(title_low):
        return ()
    title_words = set(re.findall(r"[a-zA-Z]{3,}", title_low)) if title_low else set()
    out: List[str] = []
    if isinstance(raw_keywords, (list, tuple)):
        for k in raw_keywords:
            if not isinstance(k, str) or not k.strip():
                continue
            k_clean = re.sub(r"\s+", " ", k.strip()).lower()
            if title_low and not re.search(r"\b" + re.escape(k_clean) + r"\b", title_low):
                continue
            if _is_excluded_non_crisis_attack_keyword(k_clean, title_words, crisis_stems):
                continue
            if (
                k_clean in seed_set
                or k_clean in dyn_acute_set
                or (
                    not _is_noise_dynamic_keyword(k_clean, sw)
                    and _dynamic_keyword_has_crisis_stem(k_clean, crisis_stems)
                )
            ):
                if k.strip() not in out:
                    out.append(k.strip())
    if not out and isinstance(title, str) and title.strip():
        for kw in match_high_severity_keywords(title, cache_dir=cache_dir, now_ts=now_ts):
            if kw not in out:
                out.append(kw)
    return tuple(out)


def _strip_headline_publisher_suffix(title: str) -> str:
    """Strip trailing ' - Publisher Name' suffix from RSS headline titles before dynamic n-gram discovery."""
    cleaned = (title or "").strip()
    while True:
        m = re.search(r"\s+[-|]\s+([^-|–—]+)$", cleaned)
        if not m:
            break
        prefix = cleaned[: m.start()].strip()
        suffix_raw = m.group(1).strip()
        suffix_words = suffix_raw.split()
        if len(prefix) < 12 or not suffix_words:
            break
        suffix_low = suffix_raw.lower()
        is_tagline = len(suffix_words) <= 8 and any(
            t in suffix_low for t in ("breaking news", "latest news", "live updates", ".com", ".org", ".net", ".co.")
        )
        if len(suffix_words) <= 5 or is_tagline:
            cleaned = prefix
        else:
            break
    return cleaned


def _dynamic_lexicon_path(cache_dir: Optional[str] = None) -> str:
    fname = _ssot._cfg_str("cache_files", "dynamic_lexicon_file") or "macro_lexicon_collect.json"
    return str(resolve_cachedb_filepath(os.path.join(cache_dir, fname) if cache_dir else cache_path(fname)))


def _bigram_stem_signature(term: str) -> Tuple[str, ...]:
    """Return a canonical order-independent tuple of normalized stems for a lexicon phrase."""
    tokens = [_normalize_semantic_stem(w) for w in re.findall(r"[a-z]{3,}", (term or "").lower())]
    return tuple(sorted(dict.fromkeys(t for t in tokens if t)))


def load_dynamic_lexicon(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Dict[str, Tuple[str, ...]]:
    """Load unexpired dynamically discovered high-severity and acute-shock terms from cachedb."""
    now = resolve_now_ts(now_ts)
    ttl = _ssot._cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)
    empty_res: Dict[str, Tuple[str, ...]] = {
        "high_severity_terms": (),
        "acute_shock_phrases": (),
        "dynamic_severity_keywords": (),
        "dynamic_acute_shock_phrases": (),
    }
    try:
        data = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            return empty_res
        file_sig = (_LEXICON_FILE_CACHE.get(path) or (None,))[0]
        eval_key = (path, file_sig, id(data), int(now), ttl)
        cached_eval = _LEXICON_EVAL_CACHE.get(eval_key)
        if cached_eval is not None:
            return cached_eval
        stopwords = _get_stopwords_set()
        valid_rows: List[Tuple[float, int, str, str]] = []
        for idx, (term, meta) in enumerate(entries.items()):
            if not isinstance(term, str) or not term.strip():
                continue
            t_clean = re.sub(r"\s+", " ", term.strip().lower())
            if _is_noise_dynamic_keyword(t_clean, stopwords):
                continue
            ts = normalize_epoch_sec(meta.get("ts") if isinstance(meta, dict) else meta, default=0.0) or 0.0
            tier = str(meta.get("tier", "high")).lower() if isinstance(meta, dict) else "high"
            if ts > 0.0 and (now - ts) <= ttl:
                valid_rows.append((ts, idx, t_clean, tier))
        valid_rows.sort(key=lambda r: (-r[0], r[1]))
        high_seen: Dict[str, None] = {}
        acute_seen: Dict[str, None] = {}
        high_sig_seen: Set[Tuple[str, ...]] = set()
        acute_sig_seen: Set[Tuple[str, ...]] = set()
        for _, _, t_clean, tier in valid_rows:
            sig = _bigram_stem_signature(t_clean) or (t_clean,)
            if sig not in high_sig_seen:
                high_sig_seen.add(sig)
                high_seen.setdefault(t_clean, None)
            if tier == "acute" and sig not in acute_sig_seen:
                acute_sig_seen.add(sig)
                acute_seen.setdefault(t_clean, None)
        high_tup, acute_tup = tuple(high_seen), tuple(acute_seen)
        res = {
            "high_severity_terms": high_tup,
            "acute_shock_phrases": acute_tup,
            "dynamic_severity_keywords": high_tup,
            "dynamic_acute_shock_phrases": acute_tup,
        }
        if len(_LEXICON_EVAL_CACHE) >= 32:
            _LEXICON_EVAL_CACHE.clear()
        _LEXICON_EVAL_CACHE[eval_key] = res
        return res
    except Exception as exc:
        logger.debug("Could not load dynamic lexicon from %s: %s", path, exc)
        return empty_res


def update_dynamic_lexicon(
    new_high_terms: Iterable[str],
    new_acute_terms: Iterable[str] = (),
    *,
    cache_dir: Optional[str] = None,
    source: str = "feed_discovery",
    now_ts: Optional[float] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Persist newly discovered crisis terms into ``cachedb/macro_lexicon_collect.json``."""
    now = resolve_now_ts(now_ts)
    ttl = _ssot._cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)

    stopwords = _get_stopwords_set()
    existing_entries: Dict[str, Dict[str, Any]] = {}
    sig_to_key: Dict[Tuple[str, ...], str] = {}
    pruned_existing = False
    raw = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
    if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
        raw_valid: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for idx, (k, v) in enumerate(raw["entries"].items()):
            if isinstance(k, str) and isinstance(v, dict):
                k_clean = re.sub(r"\s+", " ", k.strip().lower())
                if _is_noise_dynamic_keyword(k_clean, stopwords):
                    pruned_existing = True
                    continue
                ts = normalize_epoch_sec(v.get("ts"), default=0.0) or 0.0
                if ts > 0.0 and (now - ts) <= ttl:
                    raw_valid.append((ts, idx, k_clean, dict(v, ts=ts)))
                else:
                    pruned_existing = True
            else:
                pruned_existing = True
        raw_valid.sort(key=lambda r: (-r[0], r[1]))
        for _, _, k_clean, entry_dict in raw_valid:
            sig = _bigram_stem_signature(k_clean) or (k_clean,)
            if sig in sig_to_key:
                pruned_existing = True
                continue
            sig_to_key[sig] = k_clean
            existing_entries[k_clean] = entry_dict

    changed = pruned_existing
    for raw_t, tier_override in (
        *[(t, None) for t in new_high_terms],
        *[(a, "acute") for a in new_acute_terms],
    ):
        t = re.sub(r"\s+", " ", str(raw_t or "").strip().lower())
        if _is_noise_dynamic_keyword(t, stopwords):
            continue
        sig = _bigram_stem_signature(t) or (t,)
        prev_key = sig_to_key.get(sig, t)
        prev_entry = existing_entries.get(prev_key, {})
        prev_tier = str(prev_entry.get("tier", "high")).lower()
        prev_ts = safe_float(prev_entry.get("ts"), 0.0)
        if tier_override:
            tier = tier_override
        elif prev_tier == "acute" and prev_ts > 0.0 and (now - prev_ts) <= 3600.0:
            tier = "acute"
        else:
            tier = "high"
        if prev_key != t and prev_key in existing_entries:
            existing_entries.pop(prev_key, None)
        sig_to_key[sig] = t
        existing_entries[t] = {"ts": now, "tier": tier, "source": source}
        changed = True

    if changed:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_json_state(path, {"updated_ts": now, "entries": existing_entries}, indent=2)
        except Exception as exc:
            logger.debug("Could not save dynamic lexicon to %s: %s", path, exc)
        _LEXICON_FILE_CACHE.clear()
        _LEXICON_EVAL_CACHE.clear()

    return load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now)


def demote_dynamic_acute_terms(
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Demote dynamically discovered ``acute`` terms to ``high`` once threat is not CRITICAL_SHOCK."""
    now = resolve_now_ts(now_ts)
    ttl = _ssot._cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)
    raw = read_json_dict_cached(path, _LEXICON_FILE_CACHE)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), dict):
        return False

    stopwords = _get_stopwords_set()
    cleaned_entries: Dict[str, Dict[str, Any]] = {}
    seen_sigs: Set[Tuple[str, ...]] = set()
    raw_valid: List[Tuple[float, int, str, Dict[str, Any]]] = []
    changed = False
    for idx, (k, v) in enumerate(raw["entries"].items()):
        if not isinstance(k, str) or not isinstance(v, dict):
            changed = True
            continue
        k_clean = re.sub(r"\s+", " ", k.strip().lower())
        if _is_noise_dynamic_keyword(k_clean, stopwords):
            changed = True
            continue
        ts = normalize_epoch_sec(v.get("ts"), default=0.0) or 0.0
        if ts <= 0.0 or (now - ts) > ttl:
            changed = True
            continue
        entry = dict(v, ts=ts)
        if str(entry.get("tier", "high")).lower() == "acute":
            entry["tier"] = "high"
            changed = True
        raw_valid.append((ts, idx, k_clean, entry))
    raw_valid.sort(key=lambda r: (-r[0], r[1]))
    for _, _, k_clean, entry in raw_valid:
        sig = _bigram_stem_signature(k_clean) or (k_clean,)
        if sig in seen_sigs:
            changed = True
            continue
        seen_sigs.add(sig)
        cleaned_entries[k_clean] = entry

    if changed:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_json_state(path, {"updated_ts": now, "entries": cleaned_entries}, indent=2)
        except Exception as exc:
            logger.debug("Could not save demoted dynamic lexicon to %s: %s", path, exc)
        _LEXICON_FILE_CACHE.clear()
        _LEXICON_EVAL_CACHE.clear()
    return changed


def get_active_severity_keywords(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    seed = _split_csv(_ssot._cfg_str("lexicon_seed", "high_severity_terms"), lower=True)
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts).get("high_severity_terms", ())
    return tuple(dict.fromkeys((*seed, *dyn)))


def get_active_acute_shock_phrases(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    seed = _split_csv(_ssot._cfg_str("lexicon_seed", "acute_shock_phrases"), lower=True)
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts).get("acute_shock_phrases", ())
    return tuple(dict.fromkeys((*seed, *dyn)))


def discover_dynamic_terms_from_headlines(
    headlines: Sequence[Any],
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
    persist: bool = True,
) -> Tuple[str, ...]:
    """Extract emerging crisis bigrams/topics that surge across specialized feeds."""
    crisis_stems, stopwords = _get_crisis_stems_set(), _get_stopwords_set()
    if not crisis_stems or not headlines:
        return ()

    phrase_sources: Dict[str, Set[str]] = {}
    phrase_headlines: Dict[str, Set[str]] = {}
    phrase_counts: Dict[str, int] = {}

    phrase_has_direct_stem: Dict[str, bool] = {}
    seen_norm_titles: Set[str] = set()
    seen_canon_urls: Set[str] = set()

    for h in headlines:
        raw_title = obj_attr(h, "title", "")
        if not isinstance(raw_title, str) or not raw_title.strip():
            continue
        norm_title = re.sub(r"\s+", " ", _strip_headline_publisher_suffix(raw_title).lower()).strip()
        if not norm_title:
            continue
        canon_url = canonical_article_url(obj_attr(h, "url", ""))
        if norm_title in seen_norm_titles or (canon_url and canon_url in seen_canon_urls):
            continue
        seen_norm_titles.add(norm_title)
        if canon_url:
            seen_canon_urls.add(canon_url)
        all_words = re.findall(r"[a-zA-Z]{3,}", norm_title)
        word_set = set(all_words)
        eff_stems = set(crisis_stems)
        if word_set & _NON_CRISIS_STRIKE_CONTEXT_WORDS:
            eff_stems.discard("strike")
        if word_set & _NON_CRISIS_ATTACK_CONTEXT_WORDS:
            eff_stems.discard("attack")
        if _is_excluded_non_crisis_attack_keyword("war", word_set, crisis_stems):
            eff_stems.difference_update({"war", "escalat"})
        if _is_excluded_non_crisis_attack_keyword("liquidation", word_set, crisis_stems):
            eff_stems.difference_update({"war", "strike", "attack", "liquidat", "collaps", "crash"})
        src_key = str(obj_attr(h, "source", "") or "unknown").strip().lower()
        article_key = canon_url or norm_title
        for tokens in _extract_contiguous_token_runs(norm_title, stopwords):
            if len(tokens) < 2:
                continue
            stem_flags = [_token_matches_crisis_stem(tok, eff_stems) for tok in tokens]
            if not any(stem_flags):
                continue
            for i in range(len(tokens) - 1):
                if not any(stem_flags[max(0, i - 1) : min(len(tokens), i + 3)]):
                    continue
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                if _is_noise_dynamic_keyword(bigram, stopwords):
                    continue
                if stem_flags[i] or stem_flags[i + 1]:
                    phrase_has_direct_stem[bigram] = True
                seen_articles = phrase_headlines.setdefault(bigram, set())
                if article_key not in seen_articles:
                    seen_articles.add(article_key)
                    phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
                phrase_sources.setdefault(bigram, set()).add(src_key)

    discovered = [
        phrase
        for phrase, count in phrase_counts.items()
        if len(phrase_headlines.get(phrase, set())) >= 2
        and (
            len(phrase_sources.get(phrase, set())) >= 2
            or (count >= 2 and phrase_has_direct_stem.get(phrase, False))
        )
    ]
    if persist:
        update_dynamic_lexicon(discovered, cache_dir=cache_dir, source="specialized_feed_salience", now_ts=now_ts)
    return tuple(discovered)


def record_dynamic_crisis_terms_from_assessment(
    summary: str,
    key_risks: Sequence[str] = (),
    *,
    is_acute_shock: bool = False,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Extract salient crisis phrases from AI geopolitical/sentiment assessments into dynamic lexicon."""
    crisis_stems, stopwords = _get_crisis_stems_set(), _get_stopwords_set()
    extracted: List[str] = []
    for text in (str(summary or ""), *(str(r or "") for r in (key_risks or ()))):
        for tokens in _extract_contiguous_token_runs(text, stopwords):
            for w1, w2 in zip(tokens, tokens[1:]):
                if any(_token_matches_crisis_stem(w, crisis_stems) for w in (w1, w2)):
                    bg = f"{w1} {w2}"
                    if not _is_noise_dynamic_keyword(bg, stopwords) and bg not in extracted:
                        extracted.append(bg)
    if extracted:
        update_dynamic_lexicon(
            extracted,
            new_acute_terms=extracted if is_acute_shock else (),
            cache_dir=cache_dir,
            source="llm_assessment_feedback",
            now_ts=now_ts,
        )
    return tuple(extracted)


def build_dynamic_rss_url(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> str:
    """Dynamically construct the supplementary RSS query URL from active dynamic lexicon + config."""
    template = _ssot._cfg_str("feeds", "dynamic_search_rss_template")
    max_terms = max(4, _ssot._cfg_int("feeds", "dynamic_search_max_terms", 16))
    dyn = load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now_ts)
    candidates = (*dyn.get("acute_shock_phrases", ()), *dyn.get("high_severity_terms", ()), *get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts))
    prioritized: List[str] = []
    seen_sigs: Set[Tuple[str, ...]] = set()
    for t in candidates:
        if not t:
            continue
        sig = _bigram_stem_signature(t) or (t,)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        prioritized.append(t)
        if len(prioritized) >= max_terms:
            break
    encoded_terms = [urllib.parse.quote_plus(f'"{t}"' if " " in t else t) for t in prioritized]
    query_str = "+OR+".join(encoded_terms) if encoded_terms else "macro"
    return template.format(query=query_str) if "{query}" in template else template


def build_equity_tickers_rss_url() -> str:
    """Build the Yahoo Finance RSS headline feed URL for all active tracked equity tickers."""
    template = _ssot._cfg_str("feeds", "equity_tickers_rss_template")
    if not template:
        return ""
    try:
        from cachepaths import canonical_base_asset
        from instrument_registry import is_equity_symbol, tracked_price_instruments

        all_syms = tracked_price_instruments()[0]
        tickers = [
            base
            for sym in all_syms
            if is_equity_symbol(sym) and (base := canonical_base_asset(sym))
        ]
    except Exception:
        tickers = []
    dedup_tickers = list(dict.fromkeys(tickers))
    if not dedup_tickers:
        return ""
    return template.format(tickers=",".join(dedup_tickers)) if "{tickers}" in template else template


def get_equity_entity_keywords() -> Tuple[str, ...]:
    """Return configured equity company/ticker keywords plus active tracked equity base symbols."""
    cfg_kws = list(_split_csv(_ssot._cfg_str("feeds", "equity_entity_keywords"), lower=True))
    try:
        from cachepaths import canonical_base_asset
        from instrument_registry import is_equity_symbol, tracked_price_instruments

        for sym in tracked_price_instruments()[0]:
            if is_equity_symbol(sym):
                base = canonical_base_asset(sym).lower()
                if base and base not in cfg_kws:
                    cfg_kws.append(base)
    except Exception:
        pass
    return tuple(dict.fromkeys(k for k in cfg_kws if k))


def has_tracked_equity_entity(title: Any) -> bool:
    """Return True if ``title`` explicitly mentions any tracked equity ticker or company keyword."""
    if not isinstance(title, str) or not title.strip():
        return False
    words = set(re.findall(r"[a-z0-9]{2,}", _strip_headline_publisher_suffix(title).lower()))
    if not words:
        return False
    return bool(words & set(get_equity_entity_keywords()))


def get_specialized_rss_urls(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Tuple[str, ...]:
    """Return curated specialized editorial RSS feeds + equity ticker RSS feed + dynamically built crisis query feed."""
    dyn_url = build_dynamic_rss_url(cache_dir=cache_dir, now_ts=now_ts)
    eq_url = build_equity_tickers_rss_url()
    return tuple(
        dict.fromkeys(
            (
                *_split_csv(_ssot._cfg_str("feeds", "specialized_rss_urls")),
                *([eq_url] if eq_url else []),
                *([dyn_url] if dyn_url else []),
            )
        )
    )


def get_excluded_reference_sources() -> Tuple[str, ...]:
    """Load static encyclopedia/dictionary reference sources excluded from live breaking news."""
    return _split_csv(_ssot._cfg_str("feeds", "excluded_reference_sources"), lower=True)


def is_excluded_reference_news_item(title: Any = "", source: Any = "", url: Any = "") -> bool:
    """Return True if a headline originates from a static reference/encyclopedia domain or first-person advice column."""
    title_s = str(title or "").strip()
    title_l = title_s.lower()
    url_l = str(url or "").strip().lower()
    src_l = str(source or "").strip().lower()
    if _is_editorial_clickbait_or_advice_title(title_l):
        return True
    if url_l and ("/investingclub/" in url_l or "watch-options-action-live" in url_l):
        return True
    if ("yahoo! finance" in src_l or "feeds.finance.yahoo.com" in url_l) and title_s:
        if not has_tracked_equity_entity(title_s) and not match_high_severity_keywords(title_s):
            return True
    excluded = get_excluded_reference_sources()
    if not excluded:
        return False
    pub_sfx = ""
    if " - " in title_s:
        pub_sfx = title_s.rsplit(" - ", 1)[-1].strip().lower()
    return any(
        ref in src_l or ref in url_l or (pub_sfx and ref in pub_sfx)
        for ref in excluded
    )


def _normalize_semantic_stem(token: str) -> str:
    """Normalize common plural/verb/inflection suffixes into a canonical stem for story deduplication."""
    t = token.lower().strip()
    for sfx, repl in (
        ("interviews", "interview"),
        ("iranian", "iran"),
        ("russian", "russia"),
        ("ukrainian", "ukraine"),
        ("israeli", "israel"),
        ("hackers", "hack"),
        ("hacked", "hack"),
        ("hacking", "hack"),
        ("attacks", "attack"),
        ("attacked", "attack"),
        ("attacking", "attack"),
        ("strikes", "strike"),
        ("striking", "strike"),
        ("sanctions", "sanction"),
        ("tariffs", "tariff"),
        ("plunges", "plunge"),
        ("plunged", "plunge"),
        ("plunging", "plunge"),
        ("collapses", "collapse"),
        ("collapsed", "collapse"),
        ("collapsing", "collapse"),
    ):
        if t == sfx:
            t = repl
            break
    else:
        if len(t) > 5 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 5 and t.endswith(("ches", "shes", "xes", "zes", "sses")):
            t = t[:-2]
        elif len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
            if len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "lsz":
                t = t[:-1]
        elif len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
            if len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "lsz":
                t = t[:-1]
        elif len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "is", "us", "as", "os")):
            t = t[:-1]
    if len(t) > 4 and t.endswith("e") and not t.endswith("ee"):
        t = t[:-1]
    return t


_HEADLINE_STEMS_CACHE: Dict[Tuple[str, int], frozenset[str]] = {}
_HEADLINE_STEM_TOKEN_RE = re.compile(
    r"\$?(\d[\d,]*(?:\.\d+)?)\s*(?:bn|b|billion|m|mn|million|trillion|pct|%)?\b|[a-z]{3,}"
)


def headline_content_stems(title: Any) -> Set[str]:
    """Extract normalized content stem tokens (excluding stopwords and publisher suffixes) from a headline."""
    if not isinstance(title, str) or not title.strip():
        return set()
    stopwords = _get_stopwords_set()
    cache_key = (title, id(stopwords))
    cached = _HEADLINE_STEMS_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)
    clean = _strip_headline_publisher_suffix(title).strip().lower()
    stems: Set[str] = set()
    for m in _HEADLINE_STEM_TOKEN_RE.finditer(clean):
        num_grp = m.group(1)
        if num_grp:
            stems.add(num_grp.replace(",", ""))
            continue
        tok = m.group(0)
        if tok in stopwords:
            continue
        stem = _normalize_semantic_stem(tok)
        if len(stem) >= 3 and stem not in stopwords:
            stems.add(stem)
    if len(_HEADLINE_STEMS_CACHE) >= 2048:
        _HEADLINE_STEMS_CACHE.clear()
    _HEADLINE_STEMS_CACHE[cache_key] = frozenset(stems)
    return stems


_GENERIC_HEADLINE_OVERLAP_STEMS: frozenset[str] = frozenset({
    "after", "amid", "attack", "call", "day", "end", "first", "fresh", "global",
    "market", "new", "news", "over", "plan", "price", "report", "said", "say",
    "show", "stock", "strike", "talk", "time", "update", "war", "warn", "week",
    "world", "year",
})

_OFF_TOPIC_TABLOID_WORDS: frozenset[str] = (
    _COMMERCIAL_METAPHOR_WAR_WORDS
    | _HUMAN_INTEREST_WAR_FILLER_WORDS
    | frozenset({
        "blue", "cards", "celebrity", "cereal", "concert", "dylan", "estate",
        "exemption", "farmers", "festival", "gift", "governor", "hoax", "hoaxes",
        "irs", "jailed", "mamdani", "match", "mayor", "movie", "music", "olive",
        "player", "players", "prison", "protest", "protests", "recipe", "recipes",
        "red", "singer", "song", "storm", "subscriber", "subscribers", "weather",
        "yacht",
    })
)

_FINANCIAL_OR_GEOPOLITICAL_CONTEXT_WORDS: frozenset[str] = (
    _BROAD_MARKET_INDEX_CONTEXT_WORDS
    | _KINETIC_OR_MACRO_WAR_CONTEXT_WORDS
    | frozenset({
        "bank", "banks", "bond", "bonds", "central", "chip", "chips", "cpi",
        "currency", "debt", "dollar", "economy", "economic", "energy", "etf",
        "exports", "fed", "fomc", "gas", "gdp", "gold", "imports", "inflation",
        "interest", "investor", "investors", "jobs", "liquidity", "loan",
        "loans", "macro", "nvda", "nvidia", "opec", "payroll", "payrolls",
        "pce", "policy", "powell", "price", "prices", "rate", "rates",
        "recession", "regentis", "reserve", "retail", "rgnt", "sec",
        "semiconductor", "semiconductors", "shares", "spacex", "spcx",
        "starlink", "stock", "supply", "tariff", "tariffs", "tax", "taxes",
        "tech", "trade", "trading", "treasuries", "unemployment", "yield",
        "yields",
    })
)


_PARTISAN_MARKET_CONTEXT_WHITELIST: frozenset[str] = (
    _BROAD_MARKET_INDEX_CONTEXT_WORDS
    | frozenset({
        "bank", "banks", "bond", "bonds", "central", "cpi", "currency", "debt",
        "dollar", "economy", "economic", "etf", "fed", "fomc", "gdp", "inflation",
        "investor", "investors", "pce", "powell", "rate", "rates", "recession",
        "sec", "shares", "tariff", "tariffs", "tax", "taxes", "treasuries",
        "treasury", "yield", "yields",
    })
)

_MAJOR_GEOPOLITICAL_THEATER_WORDS: frozenset[str] = frozenset({
    "china", "chinese", "europe", "gaza", "hormuz", "iran", "iranian", "israel",
    "israeli", "korea", "korean", "lebanon", "middle", "nato", "pentagon",
    "russia", "russian", "saudi", "syria", "taiwan", "tehran", "ukraine",
    "ukrainian", "yemen",
})


def has_financial_or_geopolitical_context(title: Any) -> bool:
    """Return True if a headline contains explicit financial, market, macro, or geopolitical context tokens."""
    if not isinstance(title, str) or not title.strip() or is_excluded_reference_news_item(title):
        return False
    clean_low = _strip_headline_publisher_suffix(title).lower()
    words = set(re.findall(r"[a-z]{3,}", clean_low))
    matched_fin = words & _FINANCIAL_OR_GEOPOLITICAL_CONTEXT_WORDS
    has_multi_theater_geo = len(words & _MAJOR_GEOPOLITICAL_THEATER_WORDS) >= 2 and bool(
        words & {"aggressive", "alliance", "conflict", "cooperation", "deal", "defence", "defense", "escalation", "pact", "security", "summit", "tensions", "treaty"}
    )
    if not matched_fin and not has_multi_theater_geo:
        return False
    if matched_fin and matched_fin <= {"military", "troops"} and not (words & _MAJOR_GEOPOLITICAL_THEATER_WORDS):
        return False
    if matched_fin and matched_fin <= {"attack", "attacks"} and (words & _NON_CRISIS_ATTACK_CONTEXT_WORDS) and not (words & _MAJOR_GEOPOLITICAL_THEATER_WORDS):
        return False
    if matched_fin and matched_fin <= {"shares"} and not (
        words & {"asian", "european", "japanese", "chinese", "british", "german", "emerging", "sector", "nvda", "nvidia", "spcx", "spacex"}
    ):
        return False
    if matched_fin and matched_fin <= {"price", "prices"} and (words & {"nintendo", "switch", "playstation", "xbox", "console", "ticket", "tickets", "menu", "grocery"}):
        return False
    if matched_fin and matched_fin <= {"tax", "taxes"} and (words & {"estate", "gift", "exemption", "deduction", "refund", "irs", "filing", "retirees", "retirement"}) and not (words & _BROAD_MARKET_INDEX_CONTEXT_WORDS):
        return False
    if (words & _HUMAN_INTEREST_WAR_FILLER_WORDS) and not (words & _BROAD_MARKET_INDEX_CONTEXT_WORDS):
        return False
    if (words & {"hoax", "hoaxes", "protest", "protests"}) and not (words & _BROAD_MARKET_INDEX_CONTEXT_WORDS):
        return False
    if _is_editorial_clickbait_or_advice_title(clean_low):
        return False
    if "prediction market" in clean_low and (
        words & (_PARTISAN_ELECTION_CONTEXT_WORDS | {"democratic", "congress", "senate", "ballot", "sweep"})
    ):
        if not (words & (_PARTISAN_MARKET_CONTEXT_WHITELIST - {"market", "markets"})):
            return False
    if (words & _PARTISAN_ELECTION_CONTEXT_WORDS) and not (words & _PARTISAN_MARKET_CONTEXT_WHITELIST):
        return False
    if (words & _COMMERCIAL_METAPHOR_WAR_WORDS) and not (words & (_PARTISAN_MARKET_CONTEXT_WHITELIST | {"chip", "chips", "semiconductor", "semiconductors", "trade"})):
        return False
    return True


def is_off_topic_general_news_title(title: Any) -> bool:
    """Return True if a headline is sports/wildlife/entertainment/weather filler without macro or market relevance."""
    if not isinstance(title, str) or not title.strip():
        return False
    clean_low = _strip_headline_publisher_suffix(title).lower()
    words = set(re.findall(r"[a-z]{3,}", clean_low))
    if not words or has_financial_or_geopolitical_context(title):
        return False
    if _is_editorial_clickbait_or_advice_title(clean_low):
        return True
    return bool(words & (_SPORTS_CONTEXT_WORDS | _NON_CRISIS_ATTACK_CONTEXT_WORDS | _OFF_TOPIC_TABLOID_WORDS))


def headlines_are_semantic_duplicates(title_a: Any, title_b: Any) -> bool:
    """Return True if two headlines represent syndicated rewrites of the same underlying news story."""
    stems_a = headline_content_stems(title_a)
    if len(stems_a) < 4:
        return False
    stems_b = headline_content_stems(title_b)
    if len(stems_b) < 4:
        return False
    shared = stems_a & stems_b
    if len(shared) < 4:
        return False
    overlap = len(shared) / float(min(len(stems_a), len(stems_b)))
    if overlap >= 0.60:
        return True
    return overlap >= 0.40 and len(shared - _GENERIC_HEADLINE_OVERLAP_STEMS) >= 3


HIGH_SEVERITY_KEYWORDS: Tuple[str, ...] = ()
ACUTE_SHOCK_PHRASES: Tuple[str, ...] = ()
DEFAULT_RSS_URL: str = ""


def match_high_severity_keywords(
    text: str,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Return all matched high-severity keywords for ``text`` using the active lexicon."""
    lower = (text or "").lower()
    if not lower or _is_editorial_clickbait_or_advice_title(lower):
        return ()
    seed_set, crisis_stems = _get_seed_terms_set(), _get_crisis_stems_set()
    text_words = set(re.findall(r"[a-zA-Z]{3,}", lower))
    return tuple(
        kw
        for kw in get_active_severity_keywords(cache_dir=cache_dir, now_ts=now_ts)
        if kw in lower
        and not _is_excluded_non_crisis_attack_keyword(kw, text_words, crisis_stems)
        and (kw in seed_set or _dynamic_keyword_has_crisis_stem(kw, crisis_stems))
        and re.search(r"\b" + re.escape(kw) + r"\b", lower)
    )


def has_high_severity_keywords(
    text: str,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Return True if a headline matches any active high-severity keyword."""
    return bool(match_high_severity_keywords(text, cache_dir=cache_dir, now_ts=now_ts))


def has_acute_shock_keywords(
    texts: Sequence[Any],
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Return True if any headline in ``texts`` contains an active acute systemic shock phrase."""
    if not texts:
        return False
    phrases = get_active_acute_shock_phrases(cache_dir=cache_dir, now_ts=now_ts)
    crisis_stems = _get_crisis_stems_set()
    items = (texts,) if isinstance(texts, str) else texts
    for t in items:
        low = str(obj_attr(t, "title", None) or (t if isinstance(t, str) else "")).lower()
        if not low or _is_editorial_clickbait_or_advice_title(low):
            continue
        words = set(re.findall(r"[a-zA-Z]{3,}", low))
        if any(
            p in low and not _is_excluded_non_crisis_attack_keyword(p, words, crisis_stems)
            for p in phrases
        ):
            return True
    return False


import intelligence.macro_sentiment_ssot as _ssot  # noqa: E402

HIGH_SEVERITY_KEYWORDS = get_active_severity_keywords()
ACUTE_SHOCK_PHRASES = get_active_acute_shock_phrases()
DEFAULT_RSS_URL = build_dynamic_rss_url()
_ssot.HIGH_SEVERITY_KEYWORDS = HIGH_SEVERITY_KEYWORDS
_ssot.ACUTE_SHOCK_PHRASES = ACUTE_SHOCK_PHRASES
_ssot.DEFAULT_RSS_URL = DEFAULT_RSS_URL
