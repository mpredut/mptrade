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

logger = logging.getLogger("intelligence.macro.news_feed")

DEFAULT_RSS_URL = (
    "https://news.google.com/rss/search?"
    "q=war+OR+conflict+OR+%22military+strike%22+OR+%22energy+crisis%22+OR+%22oil+spike%22+OR+%22gas+pipeline%22+when:1d"
    "&hl=en-US&gl=US&ceid=US:en"
)

HIGH_SEVERITY_KEYWORDS = (
    "war", "military strike", "missile", "airstrike", "invasion",
    "energy crisis", "oil embargo", "strait of hormuz", "gas pipeline",
    "nuclear", "blackout", "sanctions", "escalation", "attack",
)


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


class NewsFeedCollector:
    """Collects and aggregates macro news headlines from free public RSS feeds across rolling windows."""

    def __init__(
        self,
        rss_url: str = DEFAULT_RSS_URL,
        cache_ttl_sec: float = 60.0,  # 1 minute cache for RSS fetches
        history_file: str = "cachedb/news_feed_collect.json",
        rolling_window_sec: float = 14400.0,  # 4 hours aggregation horizon
    ) -> None:
        self.rss_url = rss_url
        self.cache_ttl_sec = cache_ttl_sec
        self.history_file = history_file
        self.rolling_window_sec = rolling_window_sec
        self._cached_snapshot: Optional[NewsFeedSnapshot] = None
        self._last_fetch_ts: float = 0.0
        # Map of normalized title -> (first_seen_ts, NewsHeadline)
        self._aggregated_pool: Dict[str, Tuple[float, NewsHeadline]] = {}
        self._load_history()

    def _normalize_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", title.strip().lower())

    def _load_history(self) -> None:
        load_path = self.history_file
        if not os.path.exists(load_path) and "_collect.json" in load_path:
            legacy = load_path.replace("_collect.json", "_history.json")
            if os.path.exists(legacy):
                load_path = legacy
        if os.path.exists(load_path):
            try:
                with open(load_path, "r") as f:
                    data = json.load(f)
                now = time.time()
                for item in data:
                    ts = float(item.get("first_seen_ts", now))
                    if (now - ts) < self.rolling_window_sec:
                        h = NewsHeadline(
                            title=str(item.get("title", "")),
                            source=str(item.get("source", "")),
                            pub_date=str(item.get("pub_date", "")),
                            url=str(item.get("url", "")),
                            matched_keywords=tuple(item.get("matched_keywords", ())),
                            is_high_severity=bool(item.get("is_high_severity", False)),
                        )
                        norm = self._normalize_title(h.title)
                        if norm:
                            self._aggregated_pool[norm] = (ts, h)
            except Exception as e:
                logger.debug("Could not load news history from %s: %s", load_path, e)

    def _save_history(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
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
            with open(self.history_file, "w") as f:
                json.dump(payload, f, indent=2)
            # Dual-write legacy filename if using standardized collect naming
            if "_collect.json" in self.history_file:
                legacy = self.history_file.replace("_collect.json", "_history.json")
                with open(legacy, "w") as f:
                    json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug("Could not save news history to %s: %s", self.history_file, e)

    def _aggregate(self, latest_snapshot: NewsFeedSnapshot, now: float) -> NewsFeedSnapshot:
        """Merge latest batch into rolling pool and return aggregated snapshot."""
        # 1. Merge new headlines
        for h in latest_snapshot.headlines:
            norm = self._normalize_title(h.title)
            if norm and norm not in self._aggregated_pool:
                self._aggregated_pool[norm] = (now, h)

        # 2. Prune old headlines past rolling horizon
        cutoff = now - self.rolling_window_sec
        self._aggregated_pool = {
            k: v for k, v in self._aggregated_pool.items() if v[0] >= cutoff
        }
        self._save_history()

        # 3. Sort pool: high-severity first, then most recent first
        all_headlines = [
            v[1]
            for v in sorted(
                self._aggregated_pool.values(),
                key=lambda x: (1 if x[1].is_high_severity else 0, x[0]),
                reverse=True,
            )
        ]
        high_sev = [h for h in all_headlines if h.is_high_severity]
        has_shock = len(high_sev) >= 3 or any(
            any(w in h.title.lower() for w in ("declared war", "nuclear", "strait of hormuz", "major offensive"))
            for h in high_sev
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
        now = now_ts if now_ts is not None else time.time()
        cutoff = now - self.rolling_window_sec
        active_pool = {k: v for k, v in self._aggregated_pool.items() if v[0] >= cutoff}
        all_headlines = [
            v[1]
            for v in sorted(
                active_pool.values(),
                key=lambda x: (1 if x[1].is_high_severity else 0, x[0]),
                reverse=True,
            )
        ]
        high_sev = [h for h in all_headlines if h.is_high_severity]
        has_shock = len(high_sev) >= 3 or any(
            any(w in h.title.lower() for w in ("declared war", "nuclear", "strait of hormuz", "major offensive"))
            for h in high_sev
        )
        return NewsFeedSnapshot(
            headlines=tuple(all_headlines[:35]),
            total_fetched=len(all_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

    @staticmethod
    def parse_rss_xml(xml_bytes: bytes, now_ts: Optional[float] = None) -> Optional[NewsFeedSnapshot]:
        """Parse raw XML RSS feed and extract matched high-severity headlines."""
        if not xml_bytes:
            return None

        now = now_ts if now_ts is not None else time.time()
        try:
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            logger.warning("Failed to parse RSS XML: %s", e)
            return None

        items = root.findall("./channel/item")
        if not items:
            return None

        parsed_headlines: List[NewsHeadline] = []
        for it in items:
            title_el = it.find("title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            if not title:
                continue

            source_el = it.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else "Google News"

            date_el = it.find("pubDate")
            pub_date = date_el.text.strip() if date_el is not None and date_el.text else ""

            link_el = it.find("link")
            link = link_el.text.strip() if link_el is not None and link_el.text else ""

            lower_title = title.lower()
            matched = tuple(
                kw for kw in HIGH_SEVERITY_KEYWORDS
                if re.search(r"\b" + re.escape(kw) + r"\b", lower_title)
            )
            is_high = len(matched) > 0

            parsed_headlines.append(
                NewsHeadline(
                    title=title,
                    source=source,
                    pub_date=pub_date,
                    url=link,
                    matched_keywords=matched,
                    is_high_severity=is_high,
                )
            )

        high_sev = [h for h in parsed_headlines if h.is_high_severity]
        has_shock = len(high_sev) >= 3 or any(
            any(w in h.title.lower() for w in ("declared war", "nuclear", "strait of hormuz", "major offensive"))
            for h in high_sev
        )

        return NewsFeedSnapshot(
            headlines=tuple(parsed_headlines[:25]),
            total_fetched=len(parsed_headlines),
            high_severity_count=len(high_sev),
            has_critical_shock_keywords=has_shock,
            ts=now,
        )

    def fetch(self, force_refresh: bool = False) -> Optional[NewsFeedSnapshot]:
        """Fetch latest geopolitical RSS headlines with caching, aggregation, and error handling."""
        now = time.time()
        if not force_refresh and self._cached_snapshot and (now - self._last_fetch_ts) < self.cache_ttl_sec:
            return self._cached_snapshot

        try:
            req = urllib.request.Request(
                self.rss_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mptrade-macro/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()

            latest = self.parse_rss_xml(data, now_ts=now)
            if latest:
                aggregated = self._aggregate(latest, now)
                self._cached_snapshot = aggregated
                self._last_fetch_ts = now
                return aggregated
        except Exception as e:
            logger.warning("Failed to fetch RSS news feed: %s. Using cached fallback if available.", e)

        return self._cached_snapshot
