"""
rag/web.py — fetch one web page safely and turn it into plain text for the knowledge base.

Fetching a URL someone hands you is risky in two ways, and this file handles both:

1. The *request* can be abused (SSRF, "server-side request forgery"): a URL like
   http://127.0.0.1:8000/... or http://169.254.169.254/... makes OUR server call something inside our
   own machine or network. So before every request (and every redirect) we resolve the host name and
   refuse anything that isn't a public internet address.
2. The *response* can be abused: a huge file, or something that isn't a web page. So we cap the size,
   set a timeout, and accept only HTML, plain text and markdown.

What comes back is still untrusted *content*: the page's text is scanned for injections during ingest
(rag/ingest.py) and wrapped as untrusted when the agent reads it (tools/untrusted.py).

Known limit, fine for a local learning app: the host is resolved once for the check and again by the
HTTP client, so a hostile DNS server could answer differently the second time ("DNS rebinding").
Production code pins the checked IP address for the actual connection.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

MAX_BYTES = 2_000_000  # 2 MB: plenty for an article, stops someone feeding us a huge file
TIMEOUT_SECONDS = 10
MAX_REDIRECTS = 3
ALLOWED_TYPES = {"text/html", "text/plain", "text/markdown"}
USER_AGENT = "ArtLab/0.1 (learning project; knowledge-base ingest)"

# Page parts that are never content: scripts, styling, menus, footers, forms…
NOISE_TAGS = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"]


class UnsafeURL(ValueError):
    """The URL was refused before any request was made: wrong scheme, or a non-public address."""


class FetchError(ValueError):
    """The request was made but the response was refused: too big, wrong type, too many redirects."""


def check_url(url: str) -> None:
    """Raise UnsafeURL unless `url` is http(s) and its host resolves only to public internet addresses.

    `ip.is_global` is False for loopback (127.0.0.1, ::1), private networks (10.x, 192.168.x),
    link-local (169.254.x — where cloud metadata services live) and other reserved ranges.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURL(f"only http(s) URLs with a host are allowed: {url}")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURL(f"can't resolve {parsed.hostname}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise UnsafeURL(f"{parsed.hostname} resolves to a non-public address ({ip})")


def html_to_text(html: str) -> tuple[str, str]:
    """Turn an HTML page into (title, text).

    Steps:
      1. Remove tags that are never content (NOISE_TAGS).
      2. Turn h1–h3 headings into markdown headings ("# …", "## …"), so ingest can split the page by
         section exactly like a markdown file, and each chunk knows its heading.
      3. Take the remaining text, one line per block, dropping empty lines.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for level in (1, 2, 3):
        for heading in soup.find_all(f"h{level}"):
            heading.replace_with(f"\n{'#' * level} {heading.get_text(' ', strip=True)}\n")
    body = soup.body or soup
    lines = [line.strip() for line in body.get_text("\n").splitlines()]
    return title, "\n".join(line for line in lines if line)


def fetch_page(url: str, client: httpx.Client | None = None) -> tuple[str, str, str]:
    """Fetch `url` and return (final_url, title, text). Raises UnsafeURL or FetchError when refused.

    Args:
        client: the HTTP client to use. Tests pass one with a fake transport, so they never touch the network.

    Redirects are followed by hand (the client never follows them itself) so that every hop goes through
    `check_url`: a public page must not be able to redirect us to 127.0.0.1.
    """
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        with client.stream("GET", url, headers={"User-Agent": USER_AGENT}, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_TYPES:
                raise FetchError(f"not a text page ({content_type or 'no content type'}): {url}")

            # Read in pieces and stop as soon as the cap is passed, instead of downloading everything first.
            body = bytearray()
            for piece in response.iter_bytes():
                body += piece
                if len(body) > MAX_BYTES:
                    raise FetchError(f"page is larger than {MAX_BYTES // 1_000_000} MB: {url}")
            text = body.decode(response.encoding or "utf-8", errors="replace")

        if content_type == "text/html":
            title, text = html_to_text(text)
            return url, title or url, text
        return url, url, text
    raise FetchError(f"more than {MAX_REDIRECTS} redirects: {url}")
