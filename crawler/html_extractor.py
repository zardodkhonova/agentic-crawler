"""HTML text and link extraction via BeautifulSoup."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Comment

from crawler.url_utils import normalize_url

logger = logging.getLogger(__name__)

_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}
_WHITESPACE_RE = re.compile(r"\s+")


class HtmlExtractor:
    """Extracts clean text and hyperlinks from raw HTML."""

    def extract(self, raw_html: str, url: str) -> tuple[str, dict[str, str]]:
        """Return cleaned page text and metadata dictionary."""
        try:
            soup = BeautifulSoup(raw_html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(raw_html, "html.parser")
            except Exception as exc:
                logger.warning("HTML parse failed for %s: %s", url, exc)
                return "", {}

        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = str(meta_desc["content"]).strip()

        body = soup.find("body") or soup
        text = body.get_text(separator=" ", strip=True)
        text = _WHITESPACE_RE.sub(" ", text).strip()

        metadata = {"title": title, "description": description, "source_url": url}
        return text, metadata

    def extract_links(self, raw_html: str, base_url: str) -> list[tuple[str, str]]:
        """Extract absolute URLs and anchor text from HTML."""
        links: list[tuple[str, str]] = []
        try:
            soup = BeautifulSoup(raw_html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(raw_html, "html.parser")
            except Exception as exc:
                logger.warning("Link extraction parse failed for %s: %s", base_url, exc)
                return links

        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute = normalize_url(href, base_url)
            if not absolute:
                continue
            anchor_text = anchor.get_text(separator=" ", strip=True)
            links.append((absolute, anchor_text))
        return links
