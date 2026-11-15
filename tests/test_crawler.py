"""Crawler tests using a fake HTTP session (no network access)."""

import pytest
import requests

from src.ingestion.crawler import StripeDocCrawler


class FakeResponse:
    def __init__(self, text="", status=200, content_type="text/html; charset=utf-8"):
        self.text = text
        self.status_code = status
        self.headers = {"content-type": content_type}


def docs_page(title: str, body_words: int = 40) -> str:
    body = " ".join(f"word{i}" for i in range(body_words))
    return (
        f"<html><head><title>{title}</title></head><body>"
        f"<nav>ignore me</nav>"
        f"<main><h1>{title}</h1><p>{body}</p>"
        f"<pre><code>client.payment_intents.create()</code></pre></main>"
        f"</body></html>"
    )


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------

def test_canonicalize_url_strips_noise():
    crawler = StripeDocCrawler()

    assert (
        crawler._canonicalize_url("https://www.docs.stripe.com/payments/#anchor?utm=x")
        == "https://docs.stripe.com/payments"
    )
    assert (
        crawler._canonicalize_url("https://docs.stripe.com/payments/")
        == "https://docs.stripe.com/payments"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/payments",
        "https://docs.stripe.com/legal/terms",
        "https://docs.stripe.com/support",
        "https://docs.stripe.com/assets/logo.png",
        "https://docs.stripe.com/docs.pdf",
        "ftp://docs.stripe.com/payments",
    ],
)
def test_disallowed_urls_are_rejected(url):
    assert StripeDocCrawler()._is_allowed(url) is False


def test_allowed_urls_pass():
    crawler = StripeDocCrawler()

    assert crawler._is_allowed("https://docs.stripe.com/payments/payment-intents")
    assert crawler._is_allowed("http://docs.stripe.com/payments")


# ---------------------------------------------------------------------------
# Crawl behaviour
# ---------------------------------------------------------------------------

def test_crawl_fetches_each_page_once_and_tracks_counters(monkeypatch):
    crawler = StripeDocCrawler()
    monkeypatch.setattr("src.ingestion.crawler.time.sleep", lambda _seconds: None)

    pages = {
        "https://docs.stripe.com/": docs_page("Docs Home"),
        "https://docs.stripe.com/payments": docs_page("Payments"),
        "https://docs.stripe.com/payments/refunds": docs_page("Refunds"),
    }
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        if url not in pages:
            raise requests.HTTPError(f"404 for {url}")
        return FakeResponse(pages[url])

    monkeypatch.setattr(crawler.session, "get", fake_get)
    # Only the home page links onwards; the corpus is discovered breadth-first.
    pages["https://docs.stripe.com/"] = docs_page("Docs Home").replace(
        "</main>",
        '<a href="/payments">payments</a>'
        '<a href="/payments">duplicate anchor</a>'
        '<a href="/">self link</a>'
        '<a href="https://example.com/elsewhere">out of scope</a>'
        "</main>",
    )
    pages["https://docs.stripe.com/payments"] = docs_page("Payments").replace(
        "</main>", '<a href="/payments/refunds">refunds</a></main>'
    )

    result = crawler.crawl(max_pages=5)

    assert result.urls_crawled == 3
    assert [doc.url for doc in result.documents] == [
        "https://docs.stripe.com/",
        "https://docs.stripe.com/payments",
        "https://docs.stripe.com/payments/refunds",
    ]
    assert len(calls) == len(set(calls)) == 3  # one request per page
    assert result.urls_discovered == 3
    # The duplicate /payments anchor is collapsed at extraction, the self link
    # is skipped at the frontier, and the off-domain link is filtered out.
    assert result.urls_duplicate == 1
    assert result.urls_rejected == 0


def test_crawl_counts_failures_and_rejections(monkeypatch):
    crawler = StripeDocCrawler()
    monkeypatch.setattr("src.ingestion.crawler.time.sleep", lambda _seconds: None)

    pages = {
        "https://docs.stripe.com/": docs_page("Docs Home").replace(
            "</main>",
            '<a href="/broken">broken</a><a href="/thin">thin</a>'
            '<a href="/binary">binary</a></main>',
        ),
        "https://docs.stripe.com/thin": "<html><body><main><p>too short</p></main></body></html>",
        "https://docs.stripe.com/binary": FakeResponse(
            "{}", status=200, content_type="application/json"
        ),
    }

    def fake_get(url, timeout=None):
        if url == "https://docs.stripe.com/broken":
            raise requests.Timeout("timed out")
        page = pages.get(url)
        if isinstance(page, FakeResponse):
            return page
        if page is None:
            raise requests.HTTPError("404")
        return FakeResponse(page)

    monkeypatch.setattr(crawler.session, "get", fake_get)

    result = crawler.crawl(max_pages=5)

    assert result.urls_crawled == 1
    assert result.urls_failed == 1
    assert result.urls_rejected == 2


def test_crawl_respects_max_pages(monkeypatch):
    crawler = StripeDocCrawler()
    monkeypatch.setattr("src.ingestion.crawler.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        crawler.session,
        "get",
        lambda url, timeout=None: FakeResponse(docs_page(f"Page {url}")),
    )

    result = crawler.crawl(max_pages=1)

    assert result.urls_crawled == 1


def test_extracted_document_keeps_code_and_headings(monkeypatch):
    crawler = StripeDocCrawler()
    monkeypatch.setattr("src.ingestion.crawler.time.sleep", lambda _seconds: None)

    html = docs_page("Payment Intents")
    monkeypatch.setattr(
        crawler.session, "get", lambda url, timeout=None: FakeResponse(html)
    )

    result = crawler.crawl(max_pages=1)
    content = result.documents[0].content

    assert "# Payment Intents" in content  # heading level preserved
    assert "client.payment_intents.create()" in content
    assert "ignore me" not in content
    assert result.documents[0].url == "https://docs.stripe.com/"


def test_documents_roundtrip_through_disk(tmp_path, monkeypatch):
    from src.ingestion.crawler import load_documents

    crawler = StripeDocCrawler()
    monkeypatch.setattr("src.ingestion.crawler.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        crawler.session,
        "get",
        lambda url, timeout=None: FakeResponse(docs_page("Docs Home")),
    )
    crawler.crawl(max_pages=1)

    crawler.save_documents(str(tmp_path))
    loaded = load_documents(str(tmp_path))

    assert len(loaded) == 1
    assert loaded[0].to_dict() == crawler.documents[0].to_dict()
