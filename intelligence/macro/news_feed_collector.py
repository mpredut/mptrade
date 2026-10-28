"""Free RSS news collector monitoring breaking geopolitical, war, and energy crises."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import xml.etree.ElementTree as ET

from botcore import _parse_bool_flag
from cachepaths import cache_path
from intelligence.macro_sentiment_ssot import (
    ACUTE_SHOCK_PHRASES,
    DEFAULT_NEWS_ROLLING_WINDOW_SEC,
    DEFAULT_RSS_URL,
    HIGH_SEVERITY_KEYWORDS,
    build_dynamic_rss_url,
    discover_dynamic_terms_from_headlines,
    get_cache_filenames,
    get_specialized_rss_urls,
    has_acute_shock_keywords,
    match_high_severity_keywords,
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
DEFAULT_HISTORY_FILE = f"cachedb/{_NEWS_FILENAMES[0]}" if _NEWS_FILENAMES else "cachedb/news_feed_collect.json"


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
        ttl = 60.0
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                tf = float(cache_ttl_sec)
                if math.isfinite(tf) and tf >= 0.0:
                    ttl = tf
            except (TypeError, ValueError):
                pass
        rw = DEFAULT_NEWS_ROLLING_WINDOW_SEC
        if rolling_window_sec is not None and not isinstance(rolling_window_sec, bool):
            try:
                rf = float(rolling_window_sec)
                if math.isfinite(rf) and rf > 0.0:
                    rw = rf
            except (TypeError, ValueError):
                pass
        self.cache_ttl_sec = ttl
        self.history_file = history_file
        self.rolling_window_sec = rw
        self._cached_snapshot: Optional[NewsFeedSnapshot] = None
        self._last_fetch_ts: float = 0.0
        # Map of normalized title -> (first_seen_ts, NewsHeadline)
        self._aggregated_pool: Dict[str, Tuple[float, NewsHeadline]] = {}
        self._load_history()

    def _cache_dir(self) -> str:
        eff = self._effective_history_file()
        return os.path.dirname(os.path.abspath(eff))

    def get_active_feed_urls(self) -> Tuple[str, ...]:
        """Resolve specialized editorial RSS feeds + dynamic crisis search URL."""
        if self.rss_urls:
            return self.rss_urls
        if self._explicit_rss_url is not None:
            return (self._explicit_rss_url,)
        return get_specialized_rss_urls(cache_dir=self._cache_dir())

    def _effective_history_file(self) -> str:
        if self.history_file == DEFAULT_HISTORY_FILE:
            primary_name = _NEWS_FILENAMES[0] if _NEWS_FILENAMES else "news_feed_collect.json"
            return cache_path(primary_name)
        return self.history_file

    @property
    def _history(self) -> Dict[str, Tuple[float, NewsHeadline]]:
        """Backward-compatible alias for _aggregated_pool."""
        return self._aggregated_pool

    def _normalize_title(self, title: Any) -> str:
        if not isinstance(title, str):
            return ""
        return re.sub(r"\s+", " ", title.strip().lower())

    def _load_history(self) -> None:
        from state_io import read_recovering_previous
        hist_file = self._effective_history_file()
        candidates = [hist_file]
        if "_collect.json" in hist_file:
            candidates.append(hist_file.replace("_collect.json", "_history.json"))
        now = time.time()
        for load_path in candidates:
            try:
                data = read_recovering_previous(load_path, json.loads, default=None)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    try:
                        raw_ts = item.get("first_seen_ts", now)
                        if raw_ts is None or isinstance(raw_ts, bool):
                            continue
                        ts = float(raw_ts)
                        if not (math.isfinite(ts) and ts > 0.0):
                            continue
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= (now - ts) < self.rolling_window_sec:
                        raw_title = item.get("title")
                        if not isinstance(raw_title, str) or not raw_title.strip():
                            continue
                        kws = item.get("matched_keywords")
                        matched_kws = (
                            tuple(str(k).strip() for k in kws if isinstance(k, str) and str(k).strip())
                            if isinstance(kws, (list, tuple))
                            else ()
                        )
                        raw_src = item.get("source")
                        raw_pub = item.get("pub_date")
                        raw_url = item.get("url")
                        raw_sev = item.get("is_high_severity", False)
                        is_sev = self._as_bool_flag(raw_sev)
                        h = NewsHeadline(
                            title=raw_title.strip(),
                            source=raw_src.strip() if isinstance(raw_src, str) else "",
                            pub_date=raw_pub.strip() if isinstance(raw_pub, str) else "",
                            url=raw_url.strip() if isinstance(raw_url, str) else "",
                            matched_keywords=matched_kws,
                            is_high_severity=is_sev,
                        )
                        norm = self._normalize_title(h.title)
                        if norm:
                            self._aggregated_pool[norm] = (ts, h)
                return
            except Exception as e:
                logger.debug("Could not load news history from %s: %s", load_path, e)

    def _save_history(self) -> None:
        hist_file = self._effective_history_file()
        try:
            from state_io import save_json_state
            payload = []
            for norm, (ts, h) in self._aggregated_pool.items():
                payload.append({
                    "title": h.title,
                    "source": h.source,
                    "pub_date": h.pub_date,
                    "url": h.url,
                    "matched_keywords": list(h.matched_keywords),
                    "is_high_severity": h.is_high_severity,
                    "first_seen_ts": ts,
                })
            save_json_state(hist_file, payload, indent=2)
        except Exception as e:
            logger.debug("Could not save news history to %s: %s", hist_file, e)

    _as_bool_flag = staticmethod(_parse_bool_flag)

    def _CacheDirOrNone(self) -> Optional[str]:
        try:
            return self._cache_dir()
        except Exception:
            return None

    def _aggregate(self, latest_snapshot: NewsFeedSnapshot, now: float) -> NewsFeedSnapshot:
        """Merge latest batch into rolling pool, discover emerging crisis terms, and return aggregated snapshot."""
        cache_dir = self._CacheDirOrNone()
        # 1. Merge new headlines
        if isinstance(latest_snapshot, dict):
            raw_headlines = latest_snapshot.get("headlines")
        else:
            raw_headlines = getattr(latest_snapshot, "headlines", None) if latest_snapshot is not None else None
        if isinstance(raw_headlines, (list, tuple)):
            for h in raw_headlines:
                if isinstance(h, dict):
                    raw_title = h.get("title")
                    if not isinstance(raw_title, str) or not raw_title.strip():
                        continue
                    kws = h.get("matched_keywords")
                    matched_kws = (
                        tuple(str(k).strip() for k in kws if isinstance(k, str) and str(k).strip())
                        if isinstance(kws, (list, tuple))
                        else ()
                    )
                    h_obj = NewsHeadline(
                        title=raw_title.strip(),
                        source=str(h.get("source") or "").strip(),
                        pub_date=str(h.get("pub_date") or "").strip(),
                        url=str(h.get("url") or "").strip(),
                        matched_keywords=matched_kws,
                        is_high_severity=self._as_bool_flag(h.get("is_high_severity", len(matched_kws) > 0)),
                    )
                else:
                    raw_title = getattr(h, "title", None)
                    if not isinstance(raw_title, str):
                        continue
                    h_obj = h
                norm = self._normalize_title(raw_title)
                if norm and norm not in self._aggregated_pool:
                    self._aggregated_pool[norm] = (now, h_obj)

        # 2. Prune old headlines past rolling horizon
        cutoff = now - self.rolling_window_sec
        self._aggregated_pool = {
            k: v
            for k, v in self._aggregated_pool.items()
            if isinstance(v, tuple)
            and len(v) == 2
            and not isinstance(v[0], bool)
            and isinstance(v[0], (int, float))
            and math.isfinite(float(v[0]))
            and float(v[0]) >= cutoff
            and hasattr(v[1], "title")
        }

        # 3. Discover emerging crisis terms across aggregated pool and upgrade matching headlines
        pool_items = [v[1] for v in self._aggregated_pool.values()]
        discovered = discover_dynamic_terms_from_headlines(
            pool_items,
            cache_dir=cache_dir,
            now_ts=now,
            persist=True,
        )
        if discovered:
            for norm_key, (seen_ts, h_item) in list(self._aggregated_pool.items()):
                title_str = getattr(h_item, "title", "")
                if not isinstance(title_str, str) or not title_str:
                    continue
                updated_kws = match_high_severity_keywords(title_str, cache_dir=cache_dir)
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

        # 4. Sort pool: high-severity first, then most recent first
        all_headlines = [
            v[1]
            for v in sorted(
                self._aggregated_pool.values(),
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

    def get_pool_snapshot(self, now_ts: Optional[float] = None) -> NewsFeedSnapshot:
        """Return snapshot of currently aggregated history pool without fetching network."""
        cache_dir = self._CacheDirOrNone()
        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                nf = float(now_ts)
                if math.isfinite(nf) and nf > 0.0:
                    now = nf
            except (TypeError, ValueError):
                pass
        cutoff = now - self.rolling_window_sec
        active_pool = {
            k: v
            for k, v in self._aggregated_pool.items()
            if isinstance(v, tuple)
            and len(v) == 2
            and not isinstance(v[0], bool)
            and isinstance(v[0], (int, float))
            and math.isfinite(float(v[0]))
            and cutoff <= float(v[0]) <= (now + 5.0)
            and hasattr(v[1], "title")
            and hasattr(v[1], "is_high_severity")
        }
        all_headlines = [
            v[1]
            for v in sorted(
                active_pool.values(),
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

    @staticmethod
    def parse_rss_xml(
        xml_bytes: bytes,
        now_ts: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ) -> Optional[NewsFeedSnapshot]:
        """Parse raw XML RSS feed and extract matched high-severity + dynamically discovered crisis headlines."""
        if not xml_bytes or not isinstance(xml_bytes, (bytes, bytearray, str)):
            return None

        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                nf = float(now_ts)
                if math.isfinite(nf) and nf > 0.0:
                    now = nf
            except (TypeError, ValueError, OverflowError):
                pass
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

            source_el = it.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else channel_source

            date_el = it.find("pubDate")
            pub_date = date_el.text.strip() if date_el is not None and date_el.text else ""

            link_el = it.find("link")
            link = link_el.text.strip() if link_el is not None and link_el.text else ""

            raw_items.append({
                "title": title,
                "source": source,
                "pub_date": pub_date,
                "url": link,
            })

        if not raw_items:
            return None

        # Detect any emerging multi-source crisis terms within this batch
        emerging_terms = discover_dynamic_terms_from_headlines(
            raw_items,
            cache_dir=cache_dir,
            now_ts=now,
            persist=bool(cache_dir),
        )

        parsed_headlines: List[NewsHeadline] = []
        for item in raw_items:
            title = item["title"]
            lower_title = title.lower()
            matched_list = list(match_high_severity_keywords(title, cache_dir=cache_dir))
            for em_term in emerging_terms:
                if em_term not in matched_list and re.search(r"\b" + re.escape(em_term) + r"\b", lower_title):
                    matched_list.append(em_term)
            matched = tuple(matched_list)
            is_high = len(matched) > 0

            parsed_headlines.append(
                NewsHeadline(
                    title=title,
                    source=item["source"],
                    pub_date=item["pub_date"],
                    url=item["url"],
                    matched_keywords=matched,
                    is_high_severity=is_high,
                )
            )

        high_sev = [h for h in parsed_headlines if h.is_high_severity]
        has_shock = len(high_sev) >= 3 or has_acute_shock_keywords(
            [h.title for h in high_sev],
            cache_dir=cache_dir,
        )

        return NewsFeedSnapshot(
            headlines=tuple(parsed_headlines[:25]),
            total_fetched=len(parsed_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

    def fetch(self, force_refresh: bool = False, now_ts: Optional[float] = None) -> Optional[NewsFeedSnapshot]:
        """Fetch latest geopolitical RSS headlines across specialized feeds and dynamic crisis queries."""
        force_refresh = self._as_bool_flag(force_refresh)
        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                nf = float(now_ts)
                if math.isfinite(nf) and nf > 0.0:
                    now = nf
            except (TypeError, ValueError, OverflowError):
                pass
        last_ts = 0.0
        if self._last_fetch_ts is not None and not isinstance(self._last_fetch_ts, bool):
            try:
                lf = float(self._last_fetch_ts)
                if math.isfinite(lf) and lf >= 0.0:
                    last_ts = lf
            except (TypeError, ValueError, OverflowError):
                pass
        if not force_refresh and self._cached_snapshot and last_ts > 0.0 and 0.0 <= (now - last_ts) < self.cache_ttl_sec:
            return self._cached_snapshot

        cache_dir = self._CacheDirOrNone()
        if self._explicit_rss_url is None:
            self.rss_url = build_dynamic_rss_url(cache_dir=cache_dir)

        feed_urls = self.get_active_feed_urls()
        collected_headlines: List[NewsHeadline] = []
        any_parsed = False

        for feed_url in feed_urls:
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mptrade-macro/1.0"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = resp.read()

                latest = self.parse_rss_xml(data, now_ts=now, cache_dir=cache_dir)
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
