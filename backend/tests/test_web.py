"""
Web fetch tests: only safe URLs are fetched, and only sensible pages are accepted.

No real network: URLs use a public IP *literal* (so the safety check needs no DNS lookup), and the HTTP
client gets an `httpx.MockTransport` that answers every request from the `handler` below.
"""

import httpx
import pytest

from artlab.rag.web import MAX_BYTES, FetchError, UnsafeURL, check_url, fetch_page, html_to_text

PUBLIC = "http://93.184.215.14"  # a public internet address, written as an IP so no DNS is needed

PAGE = """<html><head><title>Cable guide</title><style>.x{}</style></head>
<body><nav>Home | Shop</nav><h1>Cable guide</h1><p>Intro text.</p>
<h2>Trays</h2><p>Use a 60 cm tray.</p><script>alert(1)</script><footer>© shop</footer></body></html>"""


def client_for(handler) -> httpx.Client:
    """An HTTP client whose every request is answered by `handler(request) -> httpx.Response`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/file",  # not http(s)
        "http://localhost/admin",  # our own machine, by name
        "http://127.0.0.1:8000/api/threads",  # our own machine, by address (this very app!)
        "http://10.1.2.3/",  # private network
        "http://192.168.1.1/",  # home router
        "http://169.254.169.254/latest/meta-data",  # cloud metadata service
        "http://[::1]/",  # IPv6 loopback
    ],
)
def test_unsafe_urls_are_refused_before_any_request(url):
    """Each of these would make our server reach something it shouldn't (SSRF), so none is fetched."""
    with pytest.raises(UnsafeURL):
        check_url(url)


def test_public_url_passes_the_check():
    check_url(f"{PUBLIC}/guide")  # no exception


def test_html_becomes_plain_text_with_markdown_headings():
    """Menus, scripts, styles and footers are dropped; headings become '#' lines for the splitter."""
    title, text = html_to_text(PAGE)
    assert title == "Cable guide"
    assert text.splitlines() == ["# Cable guide", "Intro text.", "## Trays", "Use a 60 cm tray."]


def test_fetch_page_returns_title_and_text():
    client = client_for(lambda req: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE))
    url, title, text = fetch_page(f"{PUBLIC}/guide", client)
    assert (url, title) == (f"{PUBLIC}/guide", "Cable guide")
    assert "Use a 60 cm tray." in text and "alert" not in text


def test_redirect_to_a_private_address_is_refused():
    """A public page can't bounce us to 127.0.0.1: every redirect hop is checked again."""
    client = client_for(lambda req: httpx.Response(302, headers={"location": "http://127.0.0.1:8000/secret"}))
    with pytest.raises(UnsafeURL):
        fetch_page(f"{PUBLIC}/go", client)


def test_non_text_content_is_refused():
    client = client_for(lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
    with pytest.raises(FetchError, match="not a text page"):
        fetch_page(f"{PUBLIC}/file.pdf", client)


def test_oversized_page_is_refused():
    big = b"a" * (MAX_BYTES + 1)
    client = client_for(lambda req: httpx.Response(200, headers={"content-type": "text/plain"}, content=big))
    with pytest.raises(FetchError, match="larger than"):
        fetch_page(f"{PUBLIC}/huge.txt", client)
