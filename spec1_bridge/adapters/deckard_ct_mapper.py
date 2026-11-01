"""Deckard: new subdomains and shared fingerprints across new media sites.

Reads a moby-dork harvest manifest, finds hosts under watched infrastructure
domains that have not been seen before, and fingerprints the ones styled like
local news. Sites that share a Google AdSense publisher ID or a favicon may belong
to one network, which is how pink-slime local news is usually exposed.

Only DIRECT ads.txt entries count. A RESELLER line names an ad exchange's account,
which many unrelated sites list, so it says nothing about who runs a site. A shared
favicon alone is weak, because hosting and CMS defaults are shared too, so it needs
more sites before it trips. Neither proves common ownership or deception.
"""

from __future__ import annotations

import base64
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import httpx
import mmh3

from newsroom.harlan_eternity.custody import USER_AGENT
from spec1_bridge.moby_runner import ManifestError, parse_manifest_line
from spec1_bridge.sociology_telemetry import DecompositionReceipt

DETECTOR: Final = "deckard_ct_mapper"
MAX_FETCH_BYTES: Final = 1_000_000
MIN_ADSENSE_SITES: Final = 2
MIN_FAVICON_SITES: Final = 3

_HOSTNAME: Final = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_PRIVATE_SUFFIXES: Final = (".internal", ".local", ".localhost", ".test")
_MEDIA_WORDS: Final = re.compile(
    r"(news|gazette|tribune|herald|courier|chronicle|sentinel|dispatch|journal|"
    r"observer|times|daily|ledger|bulletin|examiner|reporter)"
)
_PUB_ID: Final = re.compile(r"^pub-\d{10,20}$")


@dataclass(frozen=True, slots=True)
class NewSubdomain:
    host: str
    apex: str
    first_seen: datetime
    is_media_styled: bool
    receipt: DecompositionReceipt


@dataclass(frozen=True, slots=True)
class SiteFingerprint:
    host: str
    adsense_ids: frozenset[str]
    favicon_hash: int | None
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NetworkCluster:
    signal: str  # 'adsense' or 'favicon'
    key: str
    hosts: tuple[str, ...]
    receipt: DecompositionReceipt


def _apex_for(host: str, watched: Iterable[str]) -> str | None:
    for apex in watched:
        if host.endswith(f".{apex}"):
            return apex
    return None


def is_media_styled(host: str, apex: str | None = None) -> bool:
    """Say whether a host's own labels, not its parent domain, read like a paper."""
    labels = host.removesuffix(f".{apex}") if apex else host
    return _MEDIA_WORDS.search(labels) is not None


def detect_new_subdomains(
    manifest_path: Path,
    watched_apexes: Iterable[str],
    known_hosts: Iterable[str],
) -> tuple[list[NewSubdomain], list[str]]:
    """Return unseen hosts under watched_apexes, and the lines that were rejected."""
    watched = sorted({apex.lower() for apex in watched_apexes}, key=len, reverse=True)
    seen = {host.lower() for host in known_hosts}
    found: list[NewSubdomain] = []
    rejected: list[str] = []
    with manifest_path.open(encoding="utf-8") as manifest:
        for number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                entry = parse_manifest_line(line)
            except ManifestError as error:
                rejected.append(f"line {number}: {error}")
                continue
            host = entry.domain.lower().rstrip(".")
            apex = _apex_for(host, watched)
            if apex is None or host in seen:
                continue
            seen.add(host)
            styled = is_media_styled(host, apex)
            found.append(
                NewSubdomain(
                    host,
                    apex,
                    entry.retrieved_at,
                    styled,
                    DecompositionReceipt(
                        detector=DETECTOR,
                        finding=f"{host} is a new host under {apex}.",
                        reasons=(
                            f"It appears in the harvest manifest at line {number}.",
                            "It is not in the list of hosts already known.",
                            (
                                f"It {'reads' if styled else 'does not read'} "
                                "like a news site."
                            ),
                        ),
                        caveat=(
                            "New to this ledger does not mean newly created, and a "
                            "new host says nothing about its purpose."
                        ),
                        metrics={"line": number, "apex": apex},
                    ),
                )
            )
    return found, rejected


def adsense_ids_from_ads_txt(text: str) -> frozenset[str]:
    """Return the Google publisher IDs a site declares as its own."""
    found: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]
        fields = [item.strip().lower() for item in line.split(",")]
        if len(fields) < 3 or fields[0] != "google.com" or fields[2] != "direct":
            continue
        publisher = fields[1].removeprefix("ca-")
        if _PUB_ID.match(publisher):
            found.add(publisher)
    return frozenset(found)


def favicon_hash(content: bytes) -> int:
    """Return the mmh3 hash of base64 with line breaks, as Shodan indexes it."""
    return mmh3.hash(base64.encodebytes(content))


def _is_public_hostname(host: str) -> bool:
    return bool(_HOSTNAME.match(host)) and not host.endswith(_PRIVATE_SUFFIXES)


def _fetch(client: httpx.Client, url: str) -> bytes:
    """GET a body, refusing redirects and anything over the size cap."""
    with client.stream("GET", url) as response:
        if response.status_code != httpx.codes.OK:
            raise ValueError(f"HTTP {response.status_code}")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_FETCH_BYTES:
                raise ValueError(f"over {MAX_FETCH_BYTES} bytes")
    return bytes(body)


def _looks_like_html(body: bytes) -> bool:
    return body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html"))


def fingerprint_site(host: str, client: httpx.Client) -> SiteFingerprint:
    """Fetch one site's ads.txt and favicon. Failures are recorded, never raised."""
    if not _is_public_hostname(host):
        return SiteFingerprint(host, frozenset(), None, ("not a public hostname",))

    failures: list[str] = []
    adsense: frozenset[str] = frozenset()
    icon: int | None = None
    try:
        body = _fetch(client, f"https://{host}/ads.txt")
        # A soft 404 page is not an ads.txt file.
        if not _looks_like_html(body):
            adsense = adsense_ids_from_ads_txt(body.decode("utf-8", "replace"))
    except (httpx.HTTPError, ValueError) as error:
        failures.append(f"ads.txt: {error}")
    try:
        body = _fetch(client, f"https://{host}/favicon.ico")
        if body and not _looks_like_html(body):
            icon = favicon_hash(body)
    except (httpx.HTTPError, ValueError) as error:
        failures.append(f"favicon: {error}")
    return SiteFingerprint(host, adsense, icon, tuple(failures))


def fingerprint_sites(
    hosts: Iterable[str], transport: httpx.BaseTransport | None = None
) -> list[SiteFingerprint]:
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=15.0,
        follow_redirects=False,
        transport=transport,
    ) as client:
        return [fingerprint_site(host, client) for host in sorted(set(hosts))]


def _adsense_cluster(key: str, hosts: tuple[str, ...]) -> NetworkCluster:
    return NetworkCluster(
        "adsense",
        key,
        hosts,
        DecompositionReceipt(
            detector=DETECTOR,
            finding=f"{len(hosts)} new news-styled sites share AdSense ID {key}.",
            reasons=(
                (
                    f"Each of {', '.join(hosts)} lists {key} as a DIRECT seller in "
                    "ads.txt."
                ),
                "A DIRECT entry means the site claims that ad account as its own.",
            ),
            caveat=(
                "ads.txt is self-declared and can be copied. This points to shared "
                "ad revenue, not proven ownership or deceptive intent. Check "
                "registrations and bylines before calling it a network."
            ),
            metrics={"sites": len(hosts)},
        ),
    )


def _favicon_cluster(key: int, hosts: tuple[str, ...]) -> NetworkCluster:
    return NetworkCluster(
        "favicon",
        str(key),
        hosts,
        DecompositionReceipt(
            detector=DETECTOR,
            finding=f"{len(hosts)} new news-styled sites share one favicon.",
            reasons=(
                f"{', '.join(hosts)} all serve a favicon with mmh3 hash {key}.",
                (
                    f"This signal needs at least {MIN_FAVICON_SITES} sites because it "
                    "is weak on its own."
                ),
            ),
            caveat=(
                "Website builders and hosts ship default icons, so unrelated sites "
                "can match. Treat this as a lead for checking only."
            ),
            metrics={"sites": len(hosts)},
        ),
    )


def find_network_clusters(sites: Iterable[SiteFingerprint]) -> list[NetworkCluster]:
    """Group sites by shared AdSense ID and shared favicon."""
    by_adsense: defaultdict[str, set[str]] = defaultdict(set)
    by_favicon: defaultdict[int, set[str]] = defaultdict(set)
    for site in sites:
        for publisher in site.adsense_ids:
            by_adsense[publisher].add(site.host)
        if site.favicon_hash is not None:
            by_favicon[site.favicon_hash].add(site.host)

    clusters = [
        _adsense_cluster(key, tuple(sorted(hosts)))
        for key, hosts in sorted(by_adsense.items())
        if len(hosts) >= MIN_ADSENSE_SITES
    ]
    clusters.extend(
        _favicon_cluster(key, tuple(sorted(hosts)))
        for key, hosts in sorted(by_favicon.items())
        if len(hosts) >= MIN_FAVICON_SITES
    )
    return clusters
