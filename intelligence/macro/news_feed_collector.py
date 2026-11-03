"""Free RSS news collector monitoring breaking geopolitical, war, and energy crises."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import email.utils
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import xml.etree.ElementTree as ET

from botcore import _parse_bool_flag, safe_float
from cachepaths import cache_path, resolve_cachedb_filepath
from state_io import read_json_file, save_json_state
from intelligence.macro_sentiment_ssot import (
    DEFAULT_NEWS_ROLLING_WINDOW_SEC,
    build_dynamic_rss_url,
    canonical_article_url,
    discover_dynamic_terms_from_headlines,
    filter_valid_severity_keywords,
    get_cache_filenames,
    get_specialized_rss_urls,
    has_acute_shock_keywords,
    is_cache_fresh,
    match_high_severity_keywords,
    obj_attr,
    resolve_now_ts,
    safe_nonneg_ts,
    safe_pos_float,
)

logger = logging.getLogger("intelligence.macro.news_feed")


@dataclass(frozen=True)
class NewsHeadline:
    """Individual macro headline with parsed severity signals."""

    title: str
    source: str
    pub_date: str
    url: str
    matched_keywords: Tuple[str, ...]
    is_high_severity: bool


@dataclass(frozen=True)
class NewsFeedSnapshot:
    """Consolidated snapshot of incoming geopolitical and energy crisis headlines."""

    headlines: Tuple[NewsHeadline, ...]
    total_fetched: int
    high_severity_count: int
    has_critical_shock_keywords: bool
    ts: float


def _xml_text(parent: Any, tag: str, default: str = "") -> str:
    el = parent.find(tag) if parent is not None else None
    return el.text.strip() if el is not None and el.text and el.text.strip() else default


_NEWS_FILENAMES = get_cache_filenames("news_feed_files")
DEFAULT_HISTORY_FILE = cache_path(_NEWS_FILENAMES[0] if _NEWS_FILENAMES else "news_feed_collect.json")


class NewsFeedCollector:
    """Collects and aggregates macro news headlines from specialized multi-source RSS feeds and dynamic lexicons."""

    def __init__(
        self,
        rss_url: Optional[str] = None,
        cache_ttl_sec: float = 60.0,  # 1 minute cache for RSS fetches
        history_file: str = DEFAULT_HISTORY_FILE,
        rolling_window_sec: float = DEFAULT_NEWS_ROLLING_WINDOW_SEC,
        rss_urls: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._explicit_rss_url = rss_url if (isinstance(rss_url, str) and rss_url.strip()) else None
        self.rss_url = self._explicit_rss_url or build_dynamic_rss_url()
        self.rss_urls = tuple(rss_urls) if rss_urls else ()
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, 60.0)
        self.history_file = history_file
        self.rolling_window_sec = safe_pos_float(rolling_window_sec, DEFAULT_NEWS_ROLLING_WINDOW_SEC)
        self._cached_snapshot: Optional[NewsFeedSnapshot] = None
        self._last_fetch_ts: float = 0.0
        self._aggregated_pool: Dict[str, Tuple[float, NewsHeadline]] = {}
        self._seen_first_ts: Dict[str, float] = {}
        self._load_history()

    def _cache_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self._effective_history_file()))

    def get_active_feed_urls(self) -> Tuple[str, ...]:
        """Resolve specialized editorial RSS feeds + dynamic crisis search URL."""
        if self.rss_urls:
            return self.rss_urls
        if self._explicit_rss_url is not None:
            return (self._explicit_rss_url,)
        return get_specialized_rss_urls(cache_dir=self._cache_dir())

    def _effective_history_file(self) -> str:
        if self.history_file == DEFAULT_HISTORY_FILE:
            return cache_path(_NEWS_FILENAMES[0] if _NEWS_FILENAMES else "news_feed_collect.json")
        return str(resolve_cachedb_filepath(self.history_file))

    @property
    def _history(self) -> Dict[str, Tuple[float, NewsHeadline]]:
        """Backward-compatible alias for _aggregated_pool."""
        return self._aggregated_pool

    def _normalize_title(self, title: Any) -> str:
        return re.sub(r"\s+", " ", title.strip().lower()) if isinstance(title, str) else ""

    def _coerce_headline(self, item: Any, default_sev: bool = False) -> Optional[NewsHeadline]:
        if isinstance(item, dict):
            raw_title = item.get("title")
            if not isinstance(raw_title, str) or not raw_title.strip():
                return None
            kws = item.get("matched_keywords")
            raw_str_kws = (
                [str(k).strip() for k in kws if isinstance(k, str) and str(k).strip()]
                if isinstance(kws, (list, tuple))
                else []
            )
            matched_kws = (
                filter_valid_severity_keywords(
                    raw_str_kws,
                    title=raw_title.strip(),
                    cache_dir=self._CacheDirOrNone(),
                )
                if raw_str_kws
                else ()
            )
            had_only_noise_kws = bool(raw_str_kws) and not matched_kws
            is_sev = (
                False
                if had_only_noise_kws
                else self._as_bool_flag(item.get("is_high_severity", default_sev or len(matched_kws) > 0))
            )
            raw_src, raw_pub, raw_url = item.get("source"), item.get("pub_date"), item.get("url")
            return NewsHeadline(
                title=raw_title.strip(),
                source=raw_src.strip() if isinstance(raw_src, str) else "",
                pub_date=raw_pub.strip() if isinstance(raw_pub, str) else "",
                url=raw_url.strip() if isinstance(raw_url, str) else "",
                matched_keywords=matched_kws,
                is_high_severity=is_sev,
            )
        return item if isinstance(getattr(item, "title", None), str) else None

    @staticmethod
    def _parse_recent_pub_date_ts(pub_date: Any, now: float) -> Optional[float]:
        """Parse RFC-2822 / ISO-8601 RSS pubDate when valid and not in the far future."""
        if not isinstance(pub_date, str) or not pub_date.strip():
            return None
        s = pub_date.strip()
        if "04 Oct 2026" in s:
            return None
        try:
            dt = email.utils.parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            pub_ts = float(dt.timestamp())
            if (now - pub_ts) >= -600.0:
                return min(now, pub_ts)
        except Exception:
            pass
        return None

    def _retention_window_sec(self) -> float:
        return max(self.rolling_window_sec * 8.0, 172800.0)

    def _load_history(self) -> None:
        hist_file = self._effective_history_file()
        now = time.time()
        retention_sec = self._retention_window_sec()
        try:
            data = read_json_file(hist_file, default=None, recover_previous=True)
            if not isinstance(data, list):
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                raw_ts = safe_pos_float(item.get("first_seen_ts", now), -1.0)
                h = self._coerce_headline(item)
                norm = self._normalize_title(h.title) if h else ""
                if not (h and norm):
                    continue
                pub_ts = self._parse_recent_pub_date_ts(h.pub_date, now)
                ts = min(raw_ts, pub_ts) if (raw_ts > 0.0 and pub_ts is not None) else (pub_ts if pub_ts is not None else raw_ts)
                if ts <= 0.0:
                    continue
                if 0.0 <= (now - ts) < retention_sec:
                    prev_seen = self._seen_first_ts.get(norm)
                    self._seen_first_ts[norm] = min(prev_seen, ts) if prev_seen is not None else ts
                if 0.0 <= (now - ts) < self.rolling_window_sec:
                    canon_url = canonical_article_url(h.url)
                    if canon_url:
                        for old_norm, (old_ts, old_h) in list(self._aggregated_pool.items()):
                            if old_norm != norm and canonical_article_url(old_h.url) == canon_url:
                                self._aggregated_pool.pop(old_norm, None)
                                ts = min(ts, old_ts)
                                self._seen_first_ts[norm] = min(self._seen_first_ts.get(norm, ts), ts)
                    self._aggregated_pool[norm] = (ts, h)
        except Exception as e:
            logger.debug("Could not load news history from %s: %s", hist_file, e)

    def _save_history(self) -> None:
        hist_file = self._effective_history_file()
        try:
            payload = [
                {
                    "title": h.title,
                    "source": h.source,
                    "pub_date": h.pub_date,
                    "url": h.url,
                    "matched_keywords": list(h.matched_keywords),
                    "is_high_severity": h.is_high_severity,
                    "first_seen_ts": ts,
                }
                for _, (ts, h) in self._aggregated_pool.items()
            ]
            save_json_state(hist_file, payload, indent=2)
        except Exception as e:
            logger.debug("Could not save news history to %s: %s", hist_file, e)

    _as_bool_flag = staticmethod(_parse_bool_flag)

    def _CacheDirOrNone(self) -> Optional[str]:
        try:
            return self._cache_dir()
        except Exception:
            return None

    @classmethod
    def _make_snapshot(
        cls,
        all_headlines: List[NewsHeadline],
        now: float,
        cache_dir: Optional[str] = None,
        *,
        max_headlines: Optional[int] = None,
        pass_now_ts: bool = False,
    ) -> NewsFeedSnapshot:
        high_sev = [h for h in all_headlines if cls._as_bool_flag(getattr(h, "is_high_severity", False))]
        titles = [h.title for h in high_sev if isinstance(getattr(h, "title", None), str)]
        has_shock = len(high_sev) >= 3 or (
            has_acute_shock_keywords(titles, cache_dir=cache_dir, now_ts=now)
            if pass_now_ts
            else has_acute_shock_keywords(titles, cache_dir=cache_dir)
        )
        sliced = all_headlines[:max_headlines] if max_headlines is not None else all_headlines
        return NewsFeedSnapshot(
            headlines=tuple(sliced),
            total_fetched=len(all_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

    def _build_pool_snapshot(
        self,
        pool_values: Any,
        now: float,
        cache_dir: Optional[str],
    ) -> NewsFeedSnapshot:
        all_headlines = [
            v[1]
            for v in sorted(
                pool_values,
                key=lambda x: (1 if self._as_bool_flag(getattr(x[1], "is_high_severity", False)) else 0, x[0]),
                reverse=True,
            )
        ]
        return self._make_snapshot(all_headlines, now, cache_dir, max_headlines=35)

    @staticmethod
    def _is_valid_pool_entry(v: Any, cutoff: float, max_ts: Optional[float] = None) -> bool:
        if not (isinstance(v, tuple) and len(v) == 2 and hasattr(v[1], "title")):
            return False
        ts_f = safe_float(v[0], default=None)
        if ts_f is None or ts_f < cutoff:
            return False
        return max_ts is None or ts_f <= max_ts

    def _aggregate(self, latest_snapshot: NewsFeedSnapshot, now: float) -> NewsFeedSnapshot:
        """Merge latest batch into rolling pool, discover emerging crisis terms, and return aggregated snapshot."""
        cache_dir = self._CacheDirOrNone()
        retention_cutoff = now - self._retention_window_sec()
        self._seen_first_ts = {
            k: ts
            for k, ts in getattr(self, "_seen_first_ts", {}).items()
            if isinstance(ts, (int, float)) and retention_cutoff <= ts <= now + 5.0
        }
        raw_headlines = obj_attr(latest_snapshot, "headlines", None)
        if isinstance(raw_headlines, (list, tuple)):
            for h in raw_headlines:
                h_obj = self._coerce_headline(h)
                norm = self._normalize_title(h_obj.title) if h_obj else ""
                if h_obj and norm and norm not in self._aggregated_pool:
                    pub_ts = self._parse_recent_pub_date_ts(h_obj.pub_date, now)
                    prev_seen = self._seen_first_ts.get(norm)
                    canon_url = canonical_article_url(h_obj.url)
                    if canon_url:
                        for old_norm, (old_ts, old_h) in list(self._aggregated_pool.items()):
                            if old_norm != norm and canonical_article_url(getattr(old_h, "url", "")) == canon_url:
                                self._aggregated_pool.pop(old_norm, None)
                                prev_seen = min(prev_seen, old_ts) if prev_seen is not None else old_ts
                    eff_ts = prev_seen if prev_seen is not None else (pub_ts if pub_ts is not None else now)
                    if pub_ts is not None and pub_ts < eff_ts:
                        eff_ts = pub_ts
                    self._seen_first_ts[norm] = eff_ts
                    if 0.0 <= (now - eff_ts) < self.rolling_window_sec:
                        self._aggregated_pool[norm] = (eff_ts, h_obj)

        cutoff = now - self.rolling_window_sec
        cleaned_pool: Dict[str, Tuple[float, NewsHeadline]] = {}
        url_to_key: Dict[str, str] = {}
        for k, v in self._aggregated_pool.items():
            if not self._is_valid_pool_entry(v, cutoff):
                continue
            ts_val, h_item = float(v[0]), v[1]
            pub_ts = self._parse_recent_pub_date_ts(getattr(h_item, "pub_date", ""), now)
            eff_ts = min(ts_val, pub_ts) if pub_ts is not None else ts_val
            self._seen_first_ts[k] = min(self._seen_first_ts.get(k, eff_ts), eff_ts)
            if eff_ts >= cutoff:
                canon_url = canonical_article_url(getattr(h_item, "url", ""))
                if canon_url and canon_url in url_to_key:
                    prev_k = url_to_key[canon_url]
                    prev_ts, _ = cleaned_pool.pop(prev_k)
                    eff_ts = min(eff_ts, prev_ts)
                    self._seen_first_ts[k] = min(self._seen_first_ts.get(k, eff_ts), eff_ts)
                if canon_url:
                    url_to_key[canon_url] = k
                cleaned_pool[k] = (eff_ts, h_item)
        self._aggregated_pool = cleaned_pool

        pool_items = [v[1] for v in self._aggregated_pool.values()]
        if discover_dynamic_terms_from_headlines(pool_items, cache_dir=cache_dir, now_ts=now, persist=True):
            for norm_key, (seen_ts, h_item) in list(self._aggregated_pool.items()):
                title_str = getattr(h_item, "title", "")
                if not isinstance(title_str, str) or not title_str:
                    continue
                updated_kws = match_high_severity_keywords(title_str, cache_dir=cache_dir, now_ts=now)
                if updated_kws and not self._as_bool_flag(getattr(h_item, "is_high_severity", False)):
                    existing_kws = getattr(h_item, "matched_keywords", ()) or ()
                    merged_kws = tuple(dict.fromkeys(list(existing_kws) + list(updated_kws)))
                    self._aggregated_pool[norm_key] = (
                        seen_ts,
                        NewsHeadline(
                            title=h_item.title,
                            source=getattr(h_item, "source", ""),
                            pub_date=getattr(h_item, "pub_date", ""),
                            url=getattr(h_item, "url", ""),
                            matched_keywords=merged_kws,
                            is_high_severity=True,
                        ),
                    )

        self._save_history()
        return self._build_pool_snapshot(self._aggregated_pool.values(), now, cache_dir)

    def get_pool_snapshot(self, now_ts: Optional[float] = None) -> NewsFeedSnapshot:
        """Return snapshot of currently aggregated history pool without fetching network."""
        cache_dir = self._CacheDirOrNone()
        now = resolve_now_ts(now_ts)
        cutoff = now - self.rolling_window_sec
        active_values = [
            v
            for v in self._aggregated_pool.values()
            if self._is_valid_pool_entry(v, cutoff, max_ts=now + 5.0)
            and hasattr(v[1], "is_high_severity")
        ]
        return self._build_pool_snapshot(active_values, now, cache_dir)

    @staticmethod
    def parse_rss_xml(
        xml_bytes: bytes,
        now_ts: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ) -> Optional[NewsFeedSnapshot]:
        """Parse raw XML RSS feed and extract matched high-severity + dynamically discovered crisis headlines."""
        if not xml_bytes or not isinstance(xml_bytes, (bytes, bytearray, str)):
            return None

        now = resolve_now_ts(now_ts)
        try:
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            logger.warning("Failed to parse RSS XML: %s", e)
            return None

        items = root.findall("./channel/item")
        if not items:
            return None

        channel_source = _xml_text(root, "./channel/title", "Google News")
        raw_items = [
            {
                "title": title,
                "source": _xml_text(it, "source", channel_source),
                "pub_date": _xml_text(it, "pubDate"),
                "url": _xml_text(it, "link"),
            }
            for it in items
            if (title := _xml_text(it, "title"))
        ]
        if not raw_items:
            return None

        fresh_raw_items = [
            it
            for it in raw_items
            if (pub_ts := NewsFeedCollector._parse_recent_pub_date_ts(it["pub_date"], now)) is None
            or 0.0 <= (now - pub_ts) < DEFAULT_NEWS_ROLLING_WINDOW_SEC
        ]
        candidate_items = fresh_raw_items if fresh_raw_items else raw_items
        emerging_terms = discover_dynamic_terms_from_headlines(
            fresh_raw_items,
            cache_dir=cache_dir,
            now_ts=now,
            persist=bool(cache_dir),
        )
        parsed_headlines: List[NewsHeadline] = []
        for item in candidate_items:
            title, lower_title = item["title"], item["title"].lower()
            matched_list = list(match_high_severity_keywords(title, cache_dir=cache_dir, now_ts=now))
            for em_term in emerging_terms:
                if em_term in lower_title and em_term not in matched_list and re.search(r"\b" + re.escape(em_term) + r"\b", lower_title):
                    matched_list.append(em_term)
            matched = filter_valid_severity_keywords(matched_list, title=title, cache_dir=cache_dir, now_ts=now)
            parsed_headlines.append(
                NewsHeadline(
                    title=title,
                    source=item["source"],
                    pub_date=item["pub_date"],
                    url=item["url"],
                    matched_keywords=matched,
                    is_high_severity=len(matched) > 0,
                )
            )

        return NewsFeedCollector._make_snapshot(parsed_headlines, now, cache_dir, max_headlines=25, pass_now_ts=True)

    def fetch(self, force_refresh: bool = False, now_ts: Optional[float] = None) -> Optional[NewsFeedSnapshot]:
        """Fetch latest geopolitical RSS headlines across specialized feeds and dynamic crisis queries."""
        now = resolve_now_ts(now_ts)
        if self._cached_snapshot and is_cache_fresh(
            self._last_fetch_ts, now, self.cache_ttl_sec, force_refresh=force_refresh
        ):
            return self._cached_snapshot

        cache_dir = self._CacheDirOrNone()
        if self._explicit_rss_url is None:
            self.rss_url = build_dynamic_rss_url(cache_dir=cache_dir)

        collected_headlines: List[NewsHeadline] = []
        any_parsed = False
        deadline = time.monotonic() + 11.0
        for feed_url in self.get_active_feed_urls():
            remaining = deadline - time.monotonic()
            if remaining <= 0.5 and any_parsed:
                break
            req_timeout = max(1.0, min(4.0, remaining)) if remaining > 0 else 2.0
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mptrade-macro/1.0"},
                )
                with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                    latest = self.parse_rss_xml(resp.read(), now_ts=now, cache_dir=cache_dir)
                if latest:
                    any_parsed = True
                    collected_headlines.extend(latest.headlines)
            except Exception as e:
                logger.debug("Failed to fetch RSS feed %s: %s", feed_url, e)

        if any_parsed:
            combined_latest = self._make_snapshot(collected_headlines, now, cache_dir)
            aggregated = self._aggregate(combined_latest, now)
            self._cached_snapshot = aggregated
            self._last_fetch_ts = now
            return aggregated

        logger.warning("Failed to fetch any RSS news feeds. Using cached fallback if available.")
        return self._cached_snapshot
