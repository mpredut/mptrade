"""Free RSS news collector monitoring breaking geopolitical, war, and energy crises."""
from __future__ import annotations

from dataclasses import dataclass
import logging
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
    """Collects and filters macro news headlines from free public RSS feeds without API keys."""

    def __init__(
        self,
        rss_url: str = DEFAULT_RSS_URL,
        cache_ttl_sec: float = 900.0,  # 15 minutes
    ) -> None:
        self.rss_url = rss_url
        self.cache_ttl_sec = cache_ttl_sec
        self._cached_snapshot: Optional[NewsFeedSnapshot] = None
        self._last_fetch_ts: float = 0.0

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
        """Fetch latest geopolitical RSS headlines with caching and error handling."""
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

            snapshot = self.parse_rss_xml(data, now_ts=now)
            if snapshot:
                self._cached_snapshot = snapshot
                self._last_fetch_ts = now
                return snapshot
        except Exception as e:
            logger.warning("Failed to fetch RSS news feed: %s. Using cached fallback if available.", e)

        return self._cached_snapshot
