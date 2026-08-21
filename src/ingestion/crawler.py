"""
Stripe API docs crawler and parser.
Crawls docs.stripe.com and extracts structured content.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

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


class StripeDocCrawler:
    """Crawler for Stripe API documentation."""
    
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
                    text_parts.append(f"\n## {text}\n")
                elif element.name == "pre" or element.name == "code":
                    text_parts.append(f"\n```\n{text}\n```\n")
                else:
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    
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
            
            # Only follow internal docs links
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                # Skip anchors and certain paths
                if "#" in full_url:
                    full_url = full_url.split("#")[0]
                
                # Skip non-docs paths
                path = urlparse(full_url).path
                if any(skip in path for skip in ["/support", "/contact", "/legal", ".pdf", ".png", ".jpg"]):
                    continue
                
                links.append(full_url)
        
        return list(set(links))
    
    def crawl_page(self, url: str) -> Optional[Document]:
        """Crawl a single page and extract content."""
        if url in self.visited_urls:
            return None
        
        try:
            self.visited_urls.add(url)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract content
            content = self._extract_content(soup)
            
            # Skip very short pages (likely redirects or errors)
            if len(content.split()) < 20:
                self.logger.warning(f"Skipping short page: {url}")
                return None
            
            # Extract metadata and section info
            metadata = self._extract_metadata(soup, url)
            section, subsection = self._extract_section_info(url, metadata.get("page_title", ""))
            
            # Extract title
            title = metadata.get("page_title", url.split("/")[-1].replace("-", " ").title())
            
            # Create document
            doc = Document(
                doc_id=self._generate_doc_id(url),
                url=url,
                title=title,
                content=content,
                section=section,
                subsection=subsection,
                metadata=metadata,
            )
            
            self.logger.info(f"Crawled: {title} ({len(content)} chars)")
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to crawl {url}: {e}")
            return None
    
    def crawl(self, max_pages: Optional[int] = None) -> CrawlResult:
        """Crawl Stripe docs site."""
        if max_pages is None:
            max_pages = self.config.corpus.max_pages
        
        start_time = time.time()
        urls_to_visit = [self.config.corpus.base_url]
        urls_crawled = 0
        urls_failed = 0
        
        self.logger.info(f"Starting crawl of {self.config.corpus.base_url}")
        self.logger.info(f"Max pages: {max_pages}")
        
        while urls_to_visit and urls_crawled < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            doc = self.crawl_page(url)
            
            if doc:
                self.documents.append(doc)
                urls_crawled += 1
                
                # Extract new links to visit
                try:
                    response = self.session.get(url, timeout=30)
                    soup = BeautifulSoup(response.text, "lxml")
                    new_links = self._extract_links(soup, self.config.corpus.base_url)
                    
                    for link in new_links:
                        if link not in self.visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
                except Exception:
                    pass
                
                # Rate limiting
                time.sleep(0.5)
            else:
                urls_failed += 1
        
        duration = time.time() - start_time
        
        result = CrawlResult(
            documents=self.documents,
            urls_crawled=urls_crawled,
            urls_failed=urls_failed,
            duration_seconds=duration,
            stats={
                "total_documents": len(self.documents),
                "avg_doc_length": sum(len(d.content) for d in self.documents) / max(len(self.documents), 1),
                "sections": list(set(d.section for d in self.documents)),
            },
        )
        
        self.logger.info(f"Crawl complete: {urls_crawled} pages crawled, {urls_failed} failed")
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
