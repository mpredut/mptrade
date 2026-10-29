"""Free RSS news collector monitoring breaking geopolitical, war, and energy crises."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import xml.etree.ElementTree as ET

from botcore import _parse_bool_flag, safe_float
from cachepaths import cache_path, resolve_cachedb_filepath
from intelligence.macro_sentiment_ssot import (
    DEFAULT_NEWS_ROLLING_WINDOW_SEC,
    build_dynamic_rss_url,
    discover_dynamic_terms_from_headlines,
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
            matched_kws = (
                tuple(str(k).strip() for k in kws if isinstance(k, str) and str(k).strip())
                if isinstance(kws, (list, tuple))
                else ()
            )
            raw_src, raw_pub, raw_url = item.get("source"), item.get("pub_date"), item.get("url")
            return NewsHeadline(
                title=raw_title.strip(),
                source=raw_src.strip() if isinstance(raw_src, str) else "",
                pub_date=raw_pub.strip() if isinstance(raw_pub, str) else "",
                url=raw_url.strip() if isinstance(raw_url, str) else "",
                matched_keywords=matched_kws,
                is_high_severity=self._as_bool_flag(item.get("is_high_severity", default_sev or len(matched_kws) > 0)),
            )
        return item if isinstance(getattr(item, "title", None), str) else None

    def _load_history(self) -> None:
        from state_io import read_recovering_previous
        hist_file = self._effective_history_file()
        now = time.time()
        for load_path in (hist_file,):
            try:
                data = read_recovering_previous(load_path, json.loads, default=None)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ts = safe_pos_float(item.get("first_seen_ts", now), -1.0)
                    if ts > 0.0 and 0.0 <= (now - ts) < self.rolling_window_sec:
                        h = self._coerce_headline(item)
                        norm = self._normalize_title(h.title) if h else ""
                        if h and norm:
                            self._aggregated_pool[norm] = (ts, h)
                return
            except Exception as e:
                logger.debug("Could not load news history from %s: %s", load_path, e)

    def _save_history(self) -> None:
        hist_file = self._effective_history_file()
        try:
            from state_io import save_json_state
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
        high_sev = [h for h in all_headlines if self._as_bool_flag(getattr(h, "is_high_severity", False))]
        has_shock = len(high_sev) >= 3 or has_acute_shock_keywords(
            [h.title for h in high_sev if isinstance(getattr(h, "title", None), str)],
            cache_dir=cache_dir,
        )
        return NewsFeedSnapshot(
            headlines=tuple(all_headlines[:35]),
            total_fetched=len(all_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

    def _aggregate(self, latest_snapshot: NewsFeedSnapshot, now: float) -> NewsFeedSnapshot:
        """Merge latest batch into rolling pool, discover emerging crisis terms, and return aggregated snapshot."""
        cache_dir = self._CacheDirOrNone()
        raw_headlines = obj_attr(latest_snapshot, "headlines", None)
        if isinstance(raw_headlines, (list, tuple)):
            for h in raw_headlines:
                h_obj = self._coerce_headline(h)
                norm = self._normalize_title(h_obj.title) if h_obj else ""
                if h_obj and norm and norm not in self._aggregated_pool:
                    self._aggregated_pool[norm] = (now, h_obj)

        cutoff = now - self.rolling_window_sec
        self._aggregated_pool = {
            k: v
            for k, v in self._aggregated_pool.items()
            if isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], (int, float))
            and safe_float(v[0], default=None) is not None
            and float(v[0]) >= cutoff
            and hasattr(v[1], "title")
        }

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
            if isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], (int, float))
            and safe_float(v[0], default=None) is not None
            and cutoff <= float(v[0]) <= (now + 5.0)
            and hasattr(v[1], "title")
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

        channel_title_el = root.find("./channel/title")
        channel_source = (
            channel_title_el.text.strip()
            if channel_title_el is not None and channel_title_el.text and channel_title_el.text.strip()
            else "Google News"
        )

        raw_items: List[Dict[str, str]] = []
        for it in items:
            title_el = it.find("title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            if not title:
                continue
            source_el, date_el, link_el = it.find("source"), it.find("pubDate"), it.find("link")
            raw_items.append({
                "title": title,
                "source": source_el.text.strip() if source_el is not None and source_el.text else channel_source,
                "pub_date": date_el.text.strip() if date_el is not None and date_el.text else "",
                "url": link_el.text.strip() if link_el is not None and link_el.text else "",
            })

        if not raw_items:
            return None

        emerging_terms = discover_dynamic_terms_from_headlines(raw_items, cache_dir=cache_dir, now_ts=now, persist=bool(cache_dir))
        parsed_headlines: List[NewsHeadline] = []
        for item in raw_items:
            title = item["title"]
            lower_title = title.lower()
            matched_list = list(match_high_severity_keywords(title, cache_dir=cache_dir, now_ts=now))
            for em_term in emerging_terms:
                if em_term in lower_title and em_term not in matched_list and re.search(r"\b" + re.escape(em_term) + r"\b", lower_title):
                    matched_list.append(em_term)
            matched = tuple(matched_list)
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

        high_sev = [h for h in parsed_headlines if h.is_high_severity]
        has_shock = len(high_sev) >= 3 or has_acute_shock_keywords([h.title for h in high_sev], cache_dir=cache_dir, now_ts=now)
        return NewsFeedSnapshot(
            headlines=tuple(parsed_headlines[:25]),
            total_fetched=len(parsed_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

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
            high_sev = [h for h in collected_headlines if h.is_high_severity]
            combined_latest = NewsFeedSnapshot(
                headlines=tuple(collected_headlines),
                total_fetched=len(collected_headlines),
                high_severity_count=len(high_sev),
                has_critical_shock_keywords=len(high_sev) >= 3
                or has_acute_shock_keywords([h.title for h in high_sev], cache_dir=cache_dir),
                ts=now,
            )
            aggregated = self._aggregate(combined_latest, now)
            self._cached_snapshot = aggregated
            self._last_fetch_ts = now
            return aggregated

        logger.warning("Failed to fetch any RSS news feeds. Using cached fallback if available.")
        return self._cached_snapshot
