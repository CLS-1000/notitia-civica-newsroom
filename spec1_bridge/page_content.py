"""What a sealed HTML page shows a reader, as opposed to what its markup holds.

A public-records site answers a bad link with an HTTP 200 and a page of
boilerplate: a search form, a landing page, a "Page not found" notice or a
"has not yet been released" message. The raw HTML of such a page is full of
text (scripts, styles, menus, the <head>), so checks on the raw payload pass
it. This module reads only the page's main content: the <main> element when
the page has one and it holds the content, otherwise the <body>, and never <head>, <script>, <style>,
<noscript>, <template>, <svg>, <nav>, <header>, <footer>, <aside> or elements
whose ARIA role marks them as navigation, banner or footer. <main> is
preferred only when it holds most of the page: OLIS puts just the title in it.

text_runs are the raw character data of that content exactly as it appears in
the payload, between tags and character references, leaving out link and
form-control text. Any one of them is a
verbatim span of the sealed bytes, which is what the corroboration and
grounding checks compare against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Final

from spec1_bridge.adapters.base_lazlong import normalize_whitespace

SKIPPED_TAGS: Final = frozenset(
    {
        "head",
        "title",
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "iframe",
        "object",
    }
)
SKIPPED_ROLES: Final = frozenset(
    {
        "navigation",
        "banner",
        "contentinfo",
        "search",
        "menu",
        "menubar",
        "dialog",
        "alert",
    }
)
VOID_TAGS: Final = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
# Not <form> itself: ASP.NET wraps whole pages in one, so only controls count.
FORM_TAGS: Final = frozenset({"select", "option", "label", "button", "textarea"})
FORM_CONTROLS: Final = frozenset({"input", "select", "textarea"})
HEADING_TAGS: Final = frozenset({"h1", "h2"})
# <main> is used when it holds at least this share of the body's visible text.
MAIN_SHARE: Final = 0.4
_HIDDEN_INPUT: Final = re.compile(
    r"^(hidden|submit|button|image|reset)$", re.IGNORECASE
)


@dataclass(slots=True)
class PageContent:
    """The reader-visible main content of one HTML page."""

    title: str = ""
    headings: list[str] = field(default_factory=list)
    text: str = ""
    text_runs: list[str] = field(default_factory=list)
    link_chars: int = 0
    form_chars: int = 0
    form_controls: int = 0

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class _Region:
    """Text collected from one region: the whole body, or a <main> element."""

    pieces: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    link_chars: int = 0
    form_chars: int = 0
    form_controls: int = 0


class _MainContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, skipped, main)
        self._skip_depth = 0
        self._main_depth = 0
        self._link_depth = 0
        self._form_depth = 0
        self._heading: list[str] | None = None
        self._in_title = False
        self.title: list[str] = []
        self.body = _Region()
        self.main = _Region()
        self.saw_main = False

    # Tags ----------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): (value or "") for name, value in attrs}
        if tag == "title":
            self._in_title = True
        if tag == "input" and not _HIDDEN_INPUT.match(values.get("type", "text")):
            self._count_control()
        if tag in VOID_TAGS:
            return
        skipped = (
            tag in SKIPPED_TAGS
            or values.get("role", "").lower() in SKIPPED_ROLES
            or "hidden" in values
            or values.get("aria-hidden", "").lower() == "true"
        )
        is_main = tag == "main" or values.get("role", "").lower() == "main"
        if skipped:
            self._skip_depth += 1
        if is_main and not self._skip_depth:
            self._main_depth += 1
            self.saw_main = True
        if tag == "a":
            self._link_depth += 1
        if tag in FORM_TAGS:
            self._form_depth += 1
        if tag in FORM_CONTROLS - {"input"}:
            self._count_control()
        if tag in HEADING_TAGS and not self._skip_depth:
            self._heading = []
        self._stack.append((tag, skipped, is_main))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in VOID_TAGS or not any(item[0] == tag for item in self._stack):
            return
        while self._stack:
            open_tag, skipped, is_main = self._stack.pop()
            self._close(open_tag, skipped, is_main)
            if open_tag == tag:
                break

    def _close(self, tag: str, skipped: bool, is_main: bool) -> None:
        if skipped:
            self._skip_depth -= 1
        if is_main and self._main_depth:
            self._main_depth -= 1
        if tag == "a" and self._link_depth:
            self._link_depth -= 1
        if tag in FORM_TAGS and self._form_depth:
            self._form_depth -= 1
        if tag in HEADING_TAGS and self._heading is not None:
            heading = normalize_whitespace("".join(self._heading))
            if heading:
                for region in self._regions():
                    region.headings.append(heading)
            self._heading = None

    def _count_control(self) -> None:
        if self._skip_depth:
            return
        for region in self._regions():
            region.form_controls += 1

    def _regions(self) -> list[_Region]:
        return [self.body, self.main] if self._main_depth else [self.body]

    # Text ----------------------------------------------------------------

    def _add_text(self, decoded: str, raw: str | None) -> None:
        if self._in_title:
            self.title.append(decoded)
            return
        if self._skip_depth:
            return
        if self._heading is not None:
            self._heading.append(decoded)
        size = len(decoded.strip())
        for region in self._regions():
            region.pieces.append(decoded)
            in_control = self._link_depth or self._form_depth
            if raw is not None and raw.strip() and not in_control:
                region.runs.append(raw)
            if self._link_depth:
                region.link_chars += size
            if self._form_depth:
                region.form_chars += size

    def handle_data(self, data: str) -> None:
        self._add_text(data, data)

    def handle_entityref(self, name: str) -> None:
        self._add_text(unescape(f"&{name};"), None)

    def handle_charref(self, name: str) -> None:
        self._add_text(unescape(f"&#{name};"), None)


def looks_like_html(text: str, content_type: str | None) -> bool:
    if content_type and "html" in content_type.lower():
        return True
    head = text.lstrip()[:512].lower()
    return head.startswith(("<!doctype html", "<html")) or "<body" in head


def read_main_content(html: str) -> PageContent:
    """Parse html and return its main content. Never raises on malformed markup."""
    parser = _MainContentParser()
    parser.feed(html)
    parser.close()
    region = parser.body
    if parser.saw_main:
        main_size = len(normalize_whitespace("".join(parser.main.pieces)))
        body_size = len(normalize_whitespace("".join(parser.body.pieces)))
        # OLIS wraps only the page title in <main>; the record sits after it.
        if main_size >= MAIN_SHARE * body_size:
            region = parser.main
    text = normalize_whitespace("".join(region.pieces))
    return PageContent(
        title=normalize_whitespace("".join(parser.title)),
        headings=region.headings,
        text=text,
        text_runs=[normalize_whitespace(run) for run in region.runs],
        link_chars=region.link_chars,
        form_chars=region.form_chars,
        form_controls=region.form_controls,
    )


# Notices a site serves, with HTTP 200, in place of the page that was asked for.
ERROR_NOTICES: Final = (
    re.compile(
        r"\b(page|file|document|record|measure|item|resource)s? (was |were )?not found\b"
    ),
    re.compile(r"^(error )?404\b|\b404\b.*\bnot found\b|\bnot found\b.*\b404\b"),
    re.compile(r"\b(cannot|can't|unable to) display\b"),
    re.compile(r"\b(has|have) not (yet )?been (released|published)\b"),
    re.compile(r"\b(doesn't|does not|no longer) exists?\b"),
    re.compile(r"\b(could|can) ?not be found\b|\bno longer available\b"),
    re.compile(r"\bno (results|records|matching records|filings) (were )?found\b"),
    re.compile(r"\baccess denied\b|\bsession (has )?expired\b"),
)
# A page this short is a notice, so its whole text is searched for one.
NOTICE_MAX_WORDS: Final = 120
# Fewer visible words than this and the page shows nothing.
MIN_CONTENT_WORDS: Final = 3
# A search form: input controls whose labels and options make up at least this
# share of the visible text.
SEARCH_FORM_MIN_CONTROLS: Final = 3
SEARCH_FORM_SHARE: Final = 0.4
# A landing page: most of its visible text is links elsewhere.
LANDING_LINK_SHARE: Final = 0.6


def _plain(text: str) -> str:
    return normalize_whitespace(text.replace("’", "'")).lower()


def error_notice(page: PageContent) -> str | None:
    """The not-found or not-released notice the page shows, if any.

    The title and headings are always searched. The body text is searched only
    when the page is short, so a long record that mentions "does not exist" in
    passing is not mistaken for a notice.
    """
    places = [page.title, *page.headings]
    if page.words <= NOTICE_MAX_WORDS:
        places.append(page.text)
    for place in places:
        plain = _plain(place)
        if any(pattern.search(plain) for pattern in ERROR_NOTICES):
            return place[:120]
    return None


def is_search_form_page(page: PageContent) -> bool:
    """Most of what the page shows is a form to fill in."""
    if page.form_controls < 1:
        return False
    if page.headings and _plain(page.headings[0]).startswith("search"):
        return True
    return (
        page.form_controls >= SEARCH_FORM_MIN_CONTROLS
        and page.form_chars >= SEARCH_FORM_SHARE * len(page.text)
    )


def is_link_list(page: PageContent) -> bool:
    """Most of what the page shows is links elsewhere: a landing or index page."""
    return bool(page.text) and page.link_chars >= LANDING_LINK_SHARE * len(page.text)


def hollow_page_reason(page: PageContent) -> str | None:
    """Why the page's visible content cannot be a filed record, or None."""
    if page.words < MIN_CONTENT_WORDS:
        return "shows no content"
    notice = error_notice(page)
    if notice is not None:
        return f"shows an error notice: {notice!r}"
    if is_search_form_page(page):
        return "is a search form"
    if is_link_list(page):
        return "is a landing page of links"
    return None
