from __future__ import annotations

from typing import Any

from mundialytics.data.identity import normalize_text

# Small pragmatic map for the 2026 World Cup host cities and common provider
# venue variants. This is intentionally conservative; when a venue cannot be
# resolved, the builder falls back to the user timezone and records the source.
CITY_TIMEZONES = {
    "atlanta": "America/New_York",
    "boston": "America/New_York",
    "foxborough": "America/New_York",
    "east rutherford": "America/New_York",
    "new york": "America/New_York",
    "new york new jersey": "America/New_York",
    "philadelphia": "America/New_York",
    "miami": "America/New_York",
    "miami gardens": "America/New_York",
    "toronto": "America/Toronto",
    "dallas": "America/Chicago",
    "arlington": "America/Chicago",
    "houston": "America/Chicago",
    "kansas city": "America/Chicago",
    "monterrey": "America/Monterrey",
    "guadalajara": "America/Mexico_City",
    "mexico city": "America/Mexico_City",
    "ciudad de mexico": "America/Mexico_City",
    "seattle": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "santa clara": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "inglewood": "America/Los_Angeles",
    "vancouver": "America/Vancouver",
}

COUNTRY_TIMEZONES = {
    "united states": "America/New_York",
    "usa": "America/New_York",
    "us": "America/New_York",
    "canada": "America/Toronto",
    "mexico": "America/Mexico_City",
    "world": "UTC",
}


def is_iana_timezone(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or text.upper() == "UTC":
        return bool(text)
    if "/" not in text:
        return False
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(text)
        return True
    except Exception:
        return False


def infer_event_timezone(row: dict[str, Any] | Any, *, fallback: str = "UTC") -> tuple[str, str]:
    """Infer an event-local timezone from provider venue metadata.

    Returns ``(timezone, source)``. Source is audit-friendly and can be one of:
    explicit_event_timezone, venue_city, venue_country, fixture_timezone, fallback.
    """
    get = row.get if hasattr(row, "get") else lambda k, default=None: default
    for key in ["event_timezone", "venue_timezone"]:
        val = get(key)
        if is_iana_timezone(val):
            return str(val), "explicit_event_timezone"
    city = normalize_text(get("venue_city") or get("city") or get("venue") or "")
    if city:
        for needle, tz in CITY_TIMEZONES.items():
            if needle in city:
                return tz, "venue_city"
    country = normalize_text(get("venue_country") or get("country") or get("category") or "")
    if country in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[country], "venue_country"
    ftz = get("fixture_timezone")
    if is_iana_timezone(ftz) and str(ftz).upper() != "UTC":
        return str(ftz), "fixture_timezone"
    return fallback, "fallback"
