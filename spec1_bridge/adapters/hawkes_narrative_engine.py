"""Hawkes: how a near-duplicate narrative spreads.

Posts are clustered by MinHash over 5-character shingles. Each cluster's arrival
times are fitted to a self-exciting point process:

    lambda(t) = mu + sum over t_i < t of alpha * exp(-beta * (t - t_i))

mu is the rate of arrivals not explained by earlier posts in the cluster, in posts
per hour. R_t = alpha / beta is how many further posts one post tends to trigger.

These separate outside injection from self-propagation, and no more than that. A
high mu can be bot seeding, but it is also what a news event looks like, and a high
R_t can be automated amplification as well as people sharing. The fit never labels
an account or a post as a bot, or as human. mu is fitted as a constant over the
window, so a burst that stops is averaged in.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import mmh3
import numpy as np
from psycopg import Connection
from scipy.optimize import minimize

from spec1_bridge.sociology_telemetry import (
    DOMAIN_FAKE_NEWS,
    DecompositionReceipt,
    StreamPost,
    write_observation,
)

DETECTOR: Final = "hawkes_narrative_engine"
GDELT_LAST_UPDATE_URL: Final = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

SHINGLE_SIZE: Final = 5
SIGNATURE_SIZE: Final = 64
LSH_BANDS: Final = 16
MIN_SIMILARITY: Final = 0.6
MIN_EVENTS: Final = 20
R_THRESHOLD: Final = 1.2
MU_THRESHOLD: Final = 0.8  # posts per hour
MU_TRIGGER_MIN_HOURS: Final = 24.0  # shorter windows overstate mu, so it cannot trip

# Below either gate, mu and self-excitation cannot be told apart, so none is reported.
RELIABLE_MIN_EVENTS: Final = 30
RELIABLE_MIN_HOURS: Final = 2.0
PRELIMINARY_STATUS: Final = "PRELIMINARY_BURST_OBSERVATION"
PRELIMINARY_MESSAGE: Final = (
    "Awaiting wider temporal window to reliably separate background seeding from "
    "viral excitation."
)

BETA_BOUNDS: Final = (0.1, 50.0)  # per hour; the floor stops beta collapsing to zero
_BETA_STARTS: Final = (0.5, 2.0, 10.0)
_SECONDS_PER_HOUR: Final = 3600.0
_GKG_COLUMNS: Final = 27
_GKG_DATE: Final = 1
_GKG_EXTRAS: Final = 26
_PAGE_TITLE: Final = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


def _normalized(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def shingles(text: str) -> frozenset[str]:
    """Return the overlapping 5-character pieces of text."""
    cleaned = _normalized(text)
    return frozenset(
        cleaned[index : index + SHINGLE_SIZE]
        for index in range(len(cleaned) - SHINGLE_SIZE + 1)
    )


def minhash_signature(text: str) -> tuple[int, ...] | None:
    """Return a MinHash signature, or None if text is too short to shingle."""
    pieces = shingles(text)
    if not pieces:
        return None
    return tuple(
        min(mmh3.hash(piece, seed, signed=False) for piece in pieces)
        for seed in range(SIGNATURE_SIZE)
    )


def estimated_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    """Estimate Jaccard similarity from two signatures."""
    same = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return same / len(left)


def cluster_posts(posts: Iterable[StreamPost]) -> list[list[StreamPost]]:
    """Group near-duplicate posts, largest group first."""
    signed = [
        (post, signature)
        for post in posts
        if (signature := minhash_signature(post.text)) is not None
    ]
    parents = list(range(len(signed)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    rows = SIGNATURE_SIZE // LSH_BANDS
    buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, (_, signature) in enumerate(signed):
        for band in range(LSH_BANDS):
            start = band * rows
            buckets[(band, signature[start : start + rows])].append(index)

    for members in buckets.values():
        first = members[0]
        for other in members[1:]:
            similarity = estimated_similarity(signed[first][1], signed[other][1])
            if similarity >= MIN_SIMILARITY:
                parents[find(other)] = find(first)

    groups: defaultdict[int, list[StreamPost]] = defaultdict(list)
    for index, (post, _) in enumerate(signed):
        groups[find(index)].append(post)
    return sorted(groups.values(), key=len, reverse=True)


@dataclass(frozen=True, slots=True)
class HawkesFit:
    mu: float  # posts per hour
    alpha: float
    beta: float
    events: int
    log_likelihood: float
    span_hours: float

    @property
    def r_t(self) -> float:
        return self.alpha / self.beta


def log_likelihood(
    times: Sequence[float], horizon: float, mu: float, alpha: float, beta: float
) -> float:
    """Return the Hawkes log-likelihood, using Ozaki's linear-time recursion."""
    total = -mu * horizon
    excitation = 0.0
    previous: float | None = None
    for moment in times:
        if previous is not None:
            excitation = math.exp(-beta * (moment - previous)) * (1.0 + excitation)
        total += math.log(mu + alpha * excitation)
        total -= (alpha / beta) * (1.0 - math.exp(-beta * (horizon - moment)))
        previous = moment
    return total


def fit_hawkes(times: Sequence[float], horizon: float | None = None) -> HawkesFit:
    """Fit mu, alpha, and beta by maximum likelihood. times are in hours.

    alpha may exceed beta, because a burst in progress is what this is for. Such a
    fit describes the window and does not imply growth without limit.
    """
    ordered = sorted(times)
    if len(ordered) < MIN_EVENTS:
        raise ValueError(f"a fit needs at least {MIN_EVENTS} events")
    origin = ordered[0]
    shifted = [moment - origin for moment in ordered]
    span = horizon - origin if horizon is not None else shifted[-1]
    if span <= 0:
        raise ValueError("events must span a positive length of time")

    def negative(parameters: Any) -> float:
        mu, alpha, beta = np.exp(parameters)
        return -log_likelihood(shifted, span, float(mu), float(alpha), float(beta))

    rate = len(shifted) / span
    best = None
    log_beta_bounds = (math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1]))
    for beta in _BETA_STARTS:
        start = np.log([rate / 2, beta / 2, beta])
        outcome = minimize(
            negative,
            start,
            method="L-BFGS-B",
            bounds=[(-12.0, 9.0), (-12.0, 9.0), log_beta_bounds],
        )
        if best is None or outcome.fun < best.fun:
            best = outcome
    assert best is not None
    mu, alpha, beta = (float(value) for value in np.exp(best.x))
    return HawkesFit(mu, alpha, beta, len(shifted), -float(best.fun), span)


def should_record(fit: HawkesFit) -> bool:
    """Apply the quarantine thresholds: R_t > 1.2, or mu > 0.8 posts per hour.

    mu counts only on windows of 24 hours or more. On shorter ones it reads far above
    its true value, so a burst of pure self-spread would trip it.
    """
    if fit.r_t > R_THRESHOLD:
        return True
    return fit.mu > MU_THRESHOLD and fit.span_hours >= MU_TRIGGER_MIN_HOURS


@dataclass(frozen=True, slots=True)
class NarrativeSignal:
    label: str
    representative: str
    first_seen: datetime
    fit: HawkesFit | None  # None until the window can support a fit
    receipt: DecompositionReceipt


def analyze_cluster(posts: Sequence[StreamPost]) -> NarrativeSignal | None:
    """Fit one cluster, or return None if it has too few posts to fit."""
    if len(posts) < MIN_EVENTS:
        return None
    ordered = sorted(posts, key=lambda post: post.at)
    first = ordered[0]
    hours = [
        (post.at - first.at).total_seconds() / _SECONDS_PER_HOUR for post in ordered
    ]
    if hours[-1] <= 0:
        return None
    platforms = sorted({post.platform for post in ordered})
    label = f"narrative:{mmh3.hash(_normalized(first.text), signed=False):08x}"
    finding = (
        f"{len(ordered)} near-duplicate posts over {hours[-1]:.1f} hours "
        f"on {', '.join(platforms)}."
    )
    if len(ordered) < RELIABLE_MIN_EVENTS or hours[-1] < RELIABLE_MIN_HOURS:
        return NarrativeSignal(
            label,
            first.text,
            first.at,
            None,
            DecompositionReceipt(
                detector=DETECTOR,
                finding=finding,
                reasons=(PRELIMINARY_MESSAGE,),
                caveat=(
                    f"No rates are reported below {RELIABLE_MIN_EVENTS} posts or "
                    f"{RELIABLE_MIN_HOURS} hours. This does not show who posted the "
                    "text or whether it is true."
                ),
                metrics={
                    "events": len(ordered),
                    "span_hours": hours[-1],
                    "platforms": platforms,
                },
                status=PRELIMINARY_STATUS,
            ),
        )
    fit = fit_hawkes(hours)
    tripped = []
    if fit.r_t > R_THRESHOLD:
        tripped.append(f"R_t of {fit.r_t:.2f} is above {R_THRESHOLD}.")
    if fit.mu > MU_THRESHOLD:
        if fit.span_hours >= MU_TRIGGER_MIN_HOURS:
            tripped.append(f"mu of {fit.mu:.2f} posts an hour is above {MU_THRESHOLD}.")
        else:
            tripped.append(
                f"mu of {fit.mu:.2f} posts an hour is above {MU_THRESHOLD}, but that "
                f"does not count on a window under {MU_TRIGGER_MIN_HOURS:.0f} hours, "
                "where this estimate runs high."
            )
    return NarrativeSignal(
        label,
        first.text,
        first.at,
        fit,
        DecompositionReceipt(
            detector=DETECTOR,
            finding=finding,
            reasons=(
                (
                    f"About {fit.mu:.2f} posts an hour arrived without being explained "
                    "by earlier posts in this group."
                ),
                (
                    f"Each post tended to trigger about {fit.r_t:.2f} more, and that "
                    f"effect fades with a half-life of "
                    f"{60 * math.log(2) / fit.beta:.0f} minutes."
                ),
                *tripped,
            ),
            caveat=(
                "Outside arrivals can be bots, coordination, or real news, and "
                "self-spread can be people or automation. This measures how text "
                "moved. It does not show who posted it or whether it is true."
            ),
            metrics={
                "mu": fit.mu,
                "alpha": fit.alpha,
                "beta": fit.beta,
                "r_t": fit.r_t,
                "events": fit.events,
                "span_hours": fit.span_hours,
                "platforms": platforms,
            },
        ),
    )


def write_signal(conn: Connection[Any], signal: NarrativeSignal) -> str | None:
    """Quarantine a signal that clears a threshold. Others are not written."""
    if signal.fit is None or not should_record(signal.fit):
        return None
    return write_observation(
        conn,
        domain=DOMAIN_FAKE_NEWS,
        narrative_label=signal.label,
        observed_text=signal.representative,
        receipt=signal.receipt,
        observed_at=signal.first_seen,
    )


def parse_gdelt_gkg_row(line: str) -> StreamPost | None:
    """Return a GKG 2.1 row as a post. Its text is the page title in Extras."""
    columns = line.rstrip("\n").split("\t")
    if len(columns) < _GKG_COLUMNS:
        return None
    title = _PAGE_TITLE.search(columns[_GKG_EXTRAS])
    if title is None or not title.group(1).strip():
        return None
    try:
        at = datetime.strptime(columns[_GKG_DATE], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return StreamPost("gdelt", title.group(1).strip(), at)
