"""URL normalization and domain comparison utilities."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)


def normalize_url(url: str, base_url: str | None = None) -> str:
    """Resolve relative URLs, strip fragments and common tracking parameters."""
    if base_url:
        url = urljoin(base_url, url)
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    clean = parsed._replace(
        fragment="",
        query=urlencode(filtered_query, doseq=True),
        netloc=parsed.netloc.lower(),
    )
    path = clean.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    clean = clean._replace(path=path)
    return urlunparse(clean)


def same_domain(a: str, b: str) -> bool:
    """Return True when two URLs share the same registrable domain."""
    domain_a = urlparse(a).netloc.lower().removeprefix("www.")
    domain_b = urlparse(b).netloc.lower().removeprefix("www.")
    return bool(domain_a) and domain_a == domain_b
