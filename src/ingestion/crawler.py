"""
Stripe API docs crawler and parser.
Crawls docs.stripe.com and extracts structured content.
"""

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from src.utils.config import get_config
from src.utils.logging import setup_logging, log_stage


@dataclass
class Document:
    """Represents a single document from the corpus."""
    doc_id: str
    url: str
    title: str
    content: str
    section: str = ""
    subsection: str = ""
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class CrawlResult:
    """Result of crawling operation."""
    documents: list[Document]
    urls_crawled: int
    urls_failed: int
    duration_seconds: float
    stats: dict = field(default_factory=dict)
    #: URLs added to the frontier.
    urls_discovered: int = 0
    #: URLs skipped because they were already seen or queued.
    urls_duplicate: int = 0
    #: URLs skipped because they were non-HTML, out of scope or too thin.
    urls_rejected: int = 0


class StripeDocCrawler:
    """Crawler for Stripe API documentation."""

    REQUEST_TIMEOUT = 30
    REQUEST_DELAY = 0.5
    MIN_WORDS = 20
    SKIP_EXTENSIONS = (
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".zip",
        ".css", ".js", ".json", ".xml", ".woff", ".woff2",
    )
    SKIP_PATH_FRAGMENTS = (
        "/support", "/contact", "/legal", "/login", "/signin", "/signup",
    )
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "reRankQ/1.0 (Research Project)"
        })
        self.visited_urls: set[str] = set()
        self.documents: list[Document] = []
    
    def _generate_doc_id(self, url: str) -> str:
        """Generate a stable document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _canonicalize_url(self, url: str) -> str:
        """Normalize a docs URL: drop fragments/queries, case-fold the host."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        path = parsed.path or "/"
        if len(path) > 1:
            path = path.rstrip("/") or "/"

        return urlunparse((parsed.scheme.lower() or "https", host, path, "", "", ""))

    def _base_host(self) -> str:
        host = urlparse(self.config.corpus.base_url).netloc.lower()
        return host[4:] if host.startswith("www.") else host

    def _is_allowed(self, url: str) -> bool:
        """Reject off-domain, non-http, asset and non-docs URLs."""
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host != self._base_host():
            return False

        path = parsed.path.lower()
        if path.endswith(self.SKIP_EXTENSIONS):
            return False
        if any(fragment in path for fragment in self.SKIP_PATH_FRAGMENTS):
            return False

        return True
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from a page."""
        # Remove navigation, footer, sidebar
        for element in soup.find_all(["nav", "footer", "aside", "header"]):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find("div", class_="content") or
            soup.find("div", {"role": "main"})
        )
        
        if main_content is None:
            main_content = soup.find("body") or soup
        
        # Extract text with structure
        text_parts = []
        for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "pre", "code", "li"]):
            text = element.get_text(strip=True)
            if text:
                if element.name in ["h1", "h2", "h3", "h4"]:
                    # Keep the real heading level so document structure and
                    # nesting survive extraction.
                    level = int(element.name[1])
                    text_parts.append(f"{'#' * level} {text}")
                elif element.name == "pre" or element.name == "code":
                    text_parts.append(f"```\n{text}\n```")
                else:
                    text_parts.append(text)

        # Blank lines between blocks keep paragraphs distinguishable so the
        # structure-aware chunker can split on real boundaries.
        return "\n\n".join(text_parts)
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extract metadata from the page."""
        metadata = {"url": url}
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["page_title"] = title_tag.get_text(strip=True)
        
        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            metadata["description"] = meta_desc.get("content", "")
        
        # Extract breadcrumb/navigation path
        breadcrumb = soup.find("nav", class_="breadcrumb") or soup.find("ol", class_="breadcrumb")
        if breadcrumb:
            metadata["breadcrumb"] = breadcrumb.get_text(" → ", strip=True)
        
        return metadata
    
    def _extract_section_info(self, url: str, title: str) -> tuple[str, str]:
        """Extract section and subsection from URL/title."""
        # Parse URL path for section info
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p and p != "docs"]
        
        if len(path_parts) >= 1:
            section = path_parts[0].replace("-", " ").title()
        else:
            section = "Home"
        
        if len(path_parts) >= 2:
            subsection = path_parts[1].replace("-", " ").title()
        else:
            subsection = ""
        
        return section, subsection
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract internal links from the page."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            canonical = self._canonicalize_url(full_url)

            if not self._is_allowed(canonical):
                continue

            links.append(canonical)

        return list(dict.fromkeys(links))
    
    def _document_from_soup(self, url: str, soup: BeautifulSoup) -> Optional[Document]:
        """Build a Document from an already-fetched page, or None if too thin."""
        content = self._extract_content(soup)

        # Skip very short pages (likely redirects or errors)
        if len(content.split()) < self.MIN_WORDS:
            self.logger.warning(f"Skipping short page: {url}")
            return None

        metadata = self._extract_metadata(soup, url)
        section, subsection = self._extract_section_info(url, metadata.get("page_title", ""))
        title = metadata.get("page_title", url.split("/")[-1].replace("-", " ").title())

        return Document(
            doc_id=self._generate_doc_id(url),
            url=url,
            title=title,
            content=content,
            section=section,
            subsection=subsection,
            metadata=metadata,
        )

    def crawl_page(self, url: str) -> Optional[Document]:
        """Crawl a single page and extract content."""
        if url in self.visited_urls:
            return None

        try:
            self.visited_urls.add(url)
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            doc = self._document_from_soup(url, soup)
            if doc is not None:
                self.logger.info(f"Crawled: {doc.title} ({len(doc.content)} chars)")
            return doc
        except Exception as e:
            self.logger.error(f"Failed to crawl {url}: {e}")
            return None
    
    def crawl(self, max_pages: Optional[int] = None) -> CrawlResult:
        """
        Crawl the Stripe docs site breadth-first.

        Each page is fetched exactly once. Traversal is bounded by ``max_pages``
        accepted pages and by a frontier attempt cap, so a hostile or broken
        link graph cannot keep the crawler running forever.
        """
        if max_pages is None:
            max_pages = self.config.corpus.max_pages

        start_time = time.time()
        base_url = self.config.corpus.base_url
        self.visited_urls = set()
        self.documents = []

        frontier: deque[str] = deque([self._canonicalize_url(base_url)])
        queued = set(frontier)
        discovered = len(frontier)
        duplicates = 0
        rejected = 0
        failed = 0
        max_attempts = max(max_pages * 5, max_pages)

        self.logger.info(f"Starting crawl of {base_url}")
        self.logger.info(f"Max pages: {max_pages}")

        attempts = 0
        while frontier and len(self.documents) < max_pages and attempts < max_attempts:
            url = frontier.popleft()
            attempts += 1

            if url in self.visited_urls:
                duplicates += 1
                continue
            self.visited_urls.add(url)

            try:
                response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                self.logger.error(f"Fetch failed for {url}: {exc}")
                failed += 1
                continue

            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "html" not in content_type:
                self.logger.warning(
                    f"Rejected {url} (status={response.status_code}, type={content_type or 'unknown'})"
                )
                rejected += 1
                continue

            try:
                soup = BeautifulSoup(response.text, "lxml")
            except Exception as exc:
                self.logger.warning(f"Malformed HTML at {url}: {exc}")
                rejected += 1
                continue

            try:
                doc = self._document_from_soup(url, soup)
            except Exception as exc:
                self.logger.warning(f"Content extraction failed at {url}: {exc}")
                rejected += 1
                continue

            if doc is None:
                rejected += 1
                continue

            self.documents.append(doc)
            self.logger.info(f"Crawled: {doc.title} ({len(doc.content)} chars)")

            for link in self._extract_links(soup, base_url):
                if link in self.visited_urls or link in queued:
                    duplicates += 1
                    continue
                queued.add(link)
                frontier.append(link)
                discovered += 1

            time.sleep(self.REQUEST_DELAY)

        duration = time.time() - start_time

        result = CrawlResult(
            documents=self.documents,
            urls_crawled=len(self.documents),
            urls_failed=failed,
            duration_seconds=duration,
            stats={
                "total_documents": len(self.documents),
                "avg_doc_length": sum(len(d.content) for d in self.documents) / max(len(self.documents), 1),
                "sections": list(set(d.section for d in self.documents)),
                "attempts": attempts,
            },
            urls_discovered=discovered,
            urls_duplicate=duplicates,
            urls_rejected=rejected,
        )

        self.logger.info(f"Crawl complete: {len(self.documents)} pages crawled, {failed} failed")
        self.logger.info(f"Duration: {duration:.2f}s")

        return result
    
    def save_documents(self, output_dir: str = "data/raw") -> Path:
        """Save crawled documents to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_file = output_path / "documents.jsonl"
        with open(docs_file, "w") as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict()) + "\n")
        
        # Save metadata
        meta_file = output_path / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "num_documents": len(self.documents),
                "source": self.config.corpus.base_url,
                "crawl_time": time.time(),
                "sections": list(set(d.section for d in self.documents)),
            }, f, indent=2)
        
        self.logger.info(f"Saved {len(self.documents)} documents to {output_path}")
        return docs_file


def load_documents(input_dir: str = "data/raw") -> list[Document]:
    """Load documents from disk."""
    input_path = Path(input_dir)
    docs_file = input_path / "documents.jsonl"
    
    if not docs_file.exists():
        raise FileNotFoundError(f"No documents found at {docs_file}")
    
    documents = []
    with open(docs_file, "r") as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(**data))
    
    return documents


if __name__ == "__main__":
    # Test crawler
    crawler = StripeDocCrawler()
    result = crawler.crawl(max_pages=10)  # Test with small number
    print(f"Crawled {result.urls_crawled} pages")
    print(f"Documents: {len(result.documents)}")
    
    # Save test results
    crawler.save_documents("data/raw")
