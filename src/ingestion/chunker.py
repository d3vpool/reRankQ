"""
Chunking strategies for document processing.
Implements fixed-size with overlap and structure-aware chunking.

Both strategies read their parameters from ``config.chunking``. All size
settings are expressed in tokens; fixed-size chunking converts them to
character offsets with an explicit chars-per-token factor.

Chunk integrity conventions (both strategies)
---------------------------------------------
* ``start_char``/``end_char`` index the *original* ``Document.content`` and
  always describe real document text, never synthetic text added during
  chunk construction.
* ``source_spans`` lists every original span the chunk was built from, in
  ascending order. A merged chunk covers several spans; ``start_char`` and
  ``end_char`` are then the envelope (first span start, last span end).
* ``heading_path`` holds the raw Markdown heading lines enclosing the chunk,
  root first. ``heading_context`` is the plain-text breadcrumb built from the
  ancestors of those headings.
* ``content`` is what gets embedded: the ancestor breadcrumb, then each source
  section's own heading followed by its body. Heading lines are never
  duplicated inside a chunk.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import tiktoken

from src.ingestion.crawler import Document
from src.utils.config import get_config
from src.utils.logging import setup_logging


@dataclass
class Chunk:
    """Represents a chunk of text."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    strategy: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict = field(default_factory=dict)
    #: Raw Markdown heading lines enclosing this chunk, root first.
    heading_path: list[str] = field(default_factory=list)
    #: Plain-text breadcrumb of the heading ancestors (may be empty).
    heading_context: str = ""
    #: Original document spans this chunk was built from, e.g. [[10, 240]].
    source_spans: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Section:
    """A contiguous slice of a document, tagged with its heading hierarchy."""

    heading_path: tuple[str, ...]
    body: str
    start: int
    end: int


class TextChunker:
    """Base class for text chunking strategies."""

    def __init__(self, strategy_name: str):
        self.config = get_config()
        self.logger = setup_logging()
        self.strategy_name = strategy_name
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _generate_chunk_id(self, doc_id: str, chunk_index: int, strategy: str) -> str:
        """Generate a stable chunk ID."""
        content = f"{doc_id}_{chunk_index}_{strategy}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a document. To be implemented by subclasses."""
        raise NotImplementedError


class FixedSizeChunker(TextChunker):
    """
    Fixed-size chunking with overlap.

    ``chunk_size`` and ``chunk_overlap`` are measured in tokens. Each window is
    converted to characters using ``CHARS_PER_TOKEN``, then trimmed back to a
    paragraph or sentence boundary when one falls in the second half of the
    window. Consecutive windows advance by ``(chunk_size - chunk_overlap)``
    tokens, so they overlap by roughly ``chunk_overlap`` tokens.
    """

    #: Characters per token used to translate token settings into offsets.
    #: Matches the approximation used elsewhere in the pipeline.
    CHARS_PER_TOKEN = 4

    #: Safety stop so a degenerate configuration cannot loop forever.
    MAX_CHUNKS_PER_DOCUMENT = 100_000

    def __init__(self):
        super().__init__("fixed_size")
        fixed_size = self.config.chunking.fixed_size
        self.chunk_size = fixed_size.chunk_size
        self.chunk_overlap = fixed_size.chunk_overlap
        self.effective_overlap = self._resolve_overlap()
        self.window_chars = self.chunk_size * self.CHARS_PER_TOKEN
        self.step_chars = max(
            1,
            (self.chunk_size - self.effective_overlap) * self.CHARS_PER_TOKEN,
        )

    def _resolve_overlap(self) -> int:
        """Clamp the configured overlap to a value that still makes progress."""
        if self.chunk_overlap >= self.chunk_size:
            clamped = max(self.chunk_size - 1, 0)
            self.logger.warning(
                "chunk_overlap (%d) must be smaller than chunk_size (%d); "
                "using an overlap of %d",
                self.chunk_overlap,
                self.chunk_size,
                clamped,
            )
            return clamped
        return self.chunk_overlap

    def _snap_to_boundary(self, text: str, start: int, end: int) -> int:
        """Pull the window end back to a paragraph or sentence boundary."""
        min_end = start + max(self.window_chars // 2, 1)

        for separator in ("\n\n", ". "):
            position = text.rfind(separator, start, end)
            if position >= min_end:
                return position + len(separator)

        return end

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        chunks: list[Chunk] = []
        text = doc.content
        text_length = len(text)

        if text_length == 0:
            return chunks

        start = 0

        while start < text_length and len(chunks) < self.MAX_CHUNKS_PER_DOCUMENT:
            end = min(text_length, start + self.window_chars)

            if end < text_length:
                end = self._snap_to_boundary(text, start, end)

            # Record the offsets of the stripped content so chunk metadata
            # points at exactly the text that is stored.
            raw_text = text[start:end]
            content_start = start + (len(raw_text) - len(raw_text.lstrip()))
            content_end = end - (len(raw_text) - len(raw_text.rstrip()))
            chunk_text = text[content_start:content_end]

            if chunk_text:
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=self._generate_chunk_id(
                            doc.doc_id, chunk_index, self.strategy_name
                        ),
                        doc_id=doc.doc_id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        strategy=self.strategy_name,
                        start_char=content_start,
                        end_char=content_end,
                        token_count=self.count_tokens(chunk_text),
                        metadata={
                            "section": doc.section,
                            "subsection": doc.subsection,
                            "url": doc.url,
                            "title": doc.title,
                        },
                        heading_path=[],
                        heading_context="",
                        source_spans=[[content_start, content_end]],
                    )
                )

            if end >= text_length:
                break

            # Slide forward by (chunk_size - overlap) tokens; always positive.
            start += self.step_chars

        return chunks


class StructureAwareChunker(TextChunker):
    """
    Structure-aware chunking that respects document headers.

    Markdown headings are treated as section boundaries. Each chunk keeps a
    breadcrumb of its ancestor headings plus the Markdown heading line of every
    section it contains, so an isolated chunk is still readable and searchable
    without its surrounding document. Heading text is therefore part of chunk
    content; ``source_spans`` continues to point at the original body text.

    When ``respect_headers`` is false, no heading structure is detected: the
    document is chunked purely by size (paragraph by paragraph, up to
    ``max_chunk_size``) with its Markdown left verbatim.
    """

    HEADER_PATTERN = re.compile(r"^(#{1,4})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
    PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n")

    #: Characters per token used when a unit has to be hard-split.
    #: Same approximation as the fixed-size strategy.
    CHARS_PER_TOKEN = 4

    def __init__(self):
        super().__init__("structure_aware")
        structure_aware = self.config.chunking.structure_aware
        self.max_chunk_size = structure_aware.max_chunk_size
        self.min_chunk_size = structure_aware.min_chunk_size
        self.respect_headers = structure_aware.respect_headers

    # -- section discovery ---------------------------------------------------

    @staticmethod
    def heading_level(heading: str) -> int:
        """Return the Markdown level of a raw heading line."""
        return len(heading) - len(heading.lstrip("#"))

    @staticmethod
    def plain_heading(heading: str) -> str:
        """Strip Markdown markers from a raw heading line."""
        return heading.lstrip("#").strip()

    def _trimmed_section(
        self, text: str, heading_path: tuple[str, ...], start: int, end: int
    ) -> Optional[_Section]:
        """Build a section for ``text[start:end]``, trimming surrounding blank space."""
        raw = text[start:end]
        content_start = start + (len(raw) - len(raw.lstrip()))
        content_end = end - (len(raw) - len(raw.rstrip()))
        body = text[content_start:content_end]
        if not body:
            return None
        return _Section(heading_path, body, content_start, content_end)

    def _split_sections(self, text: str) -> list[_Section]:
        """Split text into heading-tagged sections with original offsets."""
        headers = list(self.HEADER_PATTERN.finditer(text))
        sections: list[_Section] = []

        preamble_end = headers[0].start() if headers else len(text)
        preamble = self._trimmed_section(text, (), 0, preamble_end)
        if preamble is not None:
            sections.append(preamble)

        stack: list[tuple[int, str]] = []

        for index, match in enumerate(headers):
            level = len(match.group(1))
            heading = match.group(0).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))

            path = tuple(line for _, line in stack)
            body_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            section = self._trimmed_section(text, path, match.end(), body_end)
            if section is not None:
                sections.append(section)

        return sections

    # -- size handling -------------------------------------------------------

    def _paragraph_spans(self, body: str) -> list[tuple[int, int]]:
        """Offsets of non-empty paragraphs inside ``body``."""
        spans: list[tuple[int, int]] = []
        cursor = 0

        for match in self.PARAGRAPH_SEPARATOR.finditer(body):
            spans.append((cursor, match.start()))
            cursor = match.end()
        spans.append((cursor, len(body)))

        trimmed = []
        for start, end in spans:
            raw = body[start:end]
            clean_start = start + (len(raw) - len(raw.lstrip()))
            clean_end = end - (len(raw) - len(raw.rstrip()))
            if clean_end > clean_start:
                trimmed.append((clean_start, clean_end))
        return trimmed

    def _heading_cost(self, section: _Section) -> int:
        """Tokens the section's own heading line adds to the final content."""
        if not section.heading_path:
            return 0
        return self.count_tokens(section.heading_path[-1])

    def _split_oversized_unit(
        self, body: str, start: int, end: int, limit_tokens: int
    ) -> list[tuple[int, int]]:
        """Hard-split one unit that cannot be divided at any natural boundary."""
        limit = max(1, limit_tokens * self.CHARS_PER_TOKEN)
        units: list[tuple[int, int]] = []
        cursor = start

        while cursor < end:
            piece_end = min(end, cursor + limit)
            if piece_end < end:
                space = body.rfind(" ", cursor, piece_end)
                if space > cursor:
                    piece_end = space
            if piece_end <= cursor:
                piece_end = min(end, cursor + limit)

            # The character window is only an estimate; tighten it until the
            # piece really fits the configured token budget. Technical prose
            # and code tokenize far denser than 4 chars/token.
            for _ in range(8):
                piece = body[cursor:piece_end]
                tokens = self.count_tokens(piece)
                if tokens <= limit_tokens or piece_end <= cursor + 1:
                    break
                target = int(len(piece) * limit_tokens / tokens)
                piece_end = max(cursor + 1, cursor + min(target, len(piece) - 1))
                space = body.rfind(" ", cursor, piece_end)
                if space > cursor:
                    piece_end = space

            units.append((cursor, piece_end))
            cursor = piece_end
            while cursor < end and body[cursor].isspace():
                cursor += 1

        return units

    def _atomic_units(
        self, body: str, start: int, end: int, limit_tokens: int
    ) -> list[tuple[int, int]]:
        """Break a paragraph into the smallest pieces the chunker may emit."""
        if self.count_tokens(body[start:end]) <= limit_tokens:
            return [(start, end)]

        units: list[tuple[int, int]] = []
        cursor = start

        for match in re.finditer(r"(?<=[.!?])\s+", body[start:end]):
            sentence_end = start + match.start()
            if sentence_end > cursor:
                units.extend(
                    self._split_oversized_unit(body, cursor, sentence_end, limit_tokens)
                )
            cursor = start + match.end()

        if cursor < end:
            units.extend(self._split_oversized_unit(body, cursor, end, limit_tokens))

        return units

    def _split_section(self, section: _Section) -> list[_Section]:
        """Split an oversized section so each part fits ``max_chunk_size``."""
        # Reserve room for the heading line that will be prepended to the body.
        limit = max(1, self.max_chunk_size - self._heading_cost(section))

        if self.count_tokens(section.body) <= limit:
            return [section]

        units: list[tuple[int, int]] = []
        for start, end in self._paragraph_spans(section.body):
            units.extend(self._atomic_units(section.body, start, end, limit))

        parts: list[tuple[int, int]] = []
        current: Optional[tuple[int, int]] = None

        for start, end in units:
            if current is None:
                current = (start, end)
                continue
            if self.count_tokens(section.body[current[0]:end]) > limit:
                parts.append(current)
                current = (start, end)
            else:
                current = (current[0], end)

        if current is not None:
            parts.append(current)

        result: list[_Section] = []
        for start, end in parts:
            raw = section.body[start:end]
            trimmed_start = start + (len(raw) - len(raw.lstrip()))
            trimmed_end = end - (len(raw) - len(raw.rstrip()))
            if trimmed_end <= trimmed_start:
                continue
            result.append(
                _Section(
                    section.heading_path,
                    section.body[trimmed_start:trimmed_end],
                    section.start + trimmed_start,
                    section.start + trimmed_end,
                )
            )

        return result

    def _merge_sections(self, sections: list[_Section]) -> list[list[_Section]]:
        """Group consecutive sections until ``min_chunk_size`` tokens are reached."""
        groups: list[list[_Section]] = []
        current: list[_Section] = []
        current_tokens = 0

        for section in sections:
            # Heading lines are part of the final content, so they count too.
            section_tokens = self.count_tokens(section.body) + self._heading_cost(section)

            # Never let a merge push the chunk past max_chunk_size.
            if current and current_tokens + section_tokens > self.max_chunk_size:
                groups.append(current)
                current = []
                current_tokens = 0

            current.append(section)
            current_tokens += section_tokens

            if current_tokens >= self.min_chunk_size:
                groups.append(current)
                current = []
                current_tokens = 0

        if current:
            # Trailing remainder joins the previous group, as before.
            if groups:
                groups[-1].extend(current)
            else:
                groups.append(current)

        return groups

    # -- chunk construction --------------------------------------------------

    @staticmethod
    def _common_ancestors(sections: list[_Section]) -> tuple[str, ...]:
        """Longest common heading ancestry shared by every section in a group."""
        ancestors = [section.heading_path[:-1] for section in sections]
        if not ancestors:
            return ()

        shared: tuple[str, ...] = ancestors[0]
        for path in ancestors[1:]:
            limit = min(len(shared), len(path))
            cut = 0
            while cut < limit and shared[cut] == path[cut]:
                cut += 1
            shared = shared[:cut]
        return shared

    def _build_content(self, sections: list[_Section]) -> tuple[str, str]:
        """Return ``(content, heading_context)`` for a group of sections."""
        ancestors = self._common_ancestors(sections)
        breadcrumb = " > ".join(self.plain_heading(heading) for heading in ancestors)

        blocks = []
        for section in sections:
            if section.heading_path:
                blocks.append(f"{section.heading_path[-1]}\n\n{section.body}")
            else:
                blocks.append(section.body)

        body = "\n\n".join(blocks)
        content = f"{breadcrumb}\n\n{body}" if breadcrumb else body
        return content, breadcrumb

    def _build_chunk(
        self,
        doc: Document,
        sections: list[_Section],
        chunk_index: int,
    ) -> Optional[Chunk]:
        content, breadcrumb = self._build_content(sections)
        content = content.strip()
        if not content:
            return None

        spans = [[section.start, section.end] for section in sections]
        heading_path = list(sections[0].heading_path)
        own_headings = [
            section.heading_path[-1] for section in sections if section.heading_path
        ]

        metadata = {
            "section": doc.section,
            "subsection": doc.subsection,
            "header": own_headings[0] if own_headings else "",
            "heading_path": heading_path,
            "merged_headings": own_headings,
            "url": doc.url,
            "title": doc.title,
        }

        return Chunk(
            chunk_id=self._generate_chunk_id(doc.doc_id, chunk_index, self.strategy_name),
            doc_id=doc.doc_id,
            content=content,
            chunk_index=chunk_index,
            strategy=self.strategy_name,
            start_char=spans[0][0],
            end_char=spans[-1][1],
            token_count=self.count_tokens(content),
            metadata=metadata,
            heading_path=heading_path,
            heading_context=breadcrumb,
            source_spans=spans,
        )

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document respecting structure."""
        if not doc.content.strip():
            return []

        if self.respect_headers:
            sections = self._split_sections(doc.content)
        else:
            # No header detection: the whole document is one opaque section.
            body = doc.content
            stripped = body.strip()
            start = body.find(stripped)
            sections = [_Section((), stripped, start, start + len(stripped))]

        if not sections:
            return []

        split: list[_Section] = []
        for section in sections:
            split.extend(self._split_section(section))

        chunks: list[Chunk] = []
        for group in self._merge_sections(split):
            chunk = self._build_chunk(doc, group, len(chunks))
            if chunk is not None:
                chunks.append(chunk)

        return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str = "both",
    output_dir: str = "data/processed"
) -> dict[str, list[Chunk]]:
    """
    Chunk documents using specified strategy.

    Args:
        documents: List of documents to chunk
        strategy: "fixed_size", "structure_aware", or "both"
        output_dir: Directory to save chunked documents

    Returns:
        Dictionary mapping strategy name to list of chunks
    """
    logger = setup_logging()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    if strategy in ["fixed_size", "both"]:
        logger.info("Chunking with fixed-size strategy...")
        fixed_chunker = FixedSizeChunker()
        fixed_chunks = []
        for doc in documents:
            chunks = fixed_chunker.chunk_document(doc)
            fixed_chunks.extend(chunks)
        results["fixed_size"] = fixed_chunks
        logger.info(f"Fixed-size: {len(fixed_chunks)} chunks from {len(documents)} documents")

    if strategy in ["structure_aware", "both"]:
        logger.info("Chunking with structure-aware strategy...")
        structure_chunker = StructureAwareChunker()
        structure_chunks = []
        for doc in documents:
            chunks = structure_chunker.chunk_document(doc)
            structure_chunks.extend(chunks)
        results["structure_aware"] = structure_chunks
        logger.info(f"Structure-aware: {len(structure_chunks)} chunks from {len(documents)} documents")

    # Save chunks
    for strat_name, chunks in results.items():
        chunks_file = output_path / f"chunks_{strat_name}.jsonl"
        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

        # Save chunk stats
        stats = {
            "strategy": strat_name,
            "num_chunks": len(chunks),
            "avg_tokens": sum(c.token_count for c in chunks) / max(len(chunks), 1),
            "min_tokens": min(c.token_count for c in chunks) if chunks else 0,
            "max_tokens": max(c.token_count for c in chunks) if chunks else 0,
            "token_distribution": {
                "0-100": len([c for c in chunks if c.token_count < 100]),
                "100-200": len([c for c in chunks if 100 <= c.token_count < 200]),
                "200-300": len([c for c in chunks if 200 <= c.token_count < 300]),
                "300-400": len([c for c in chunks if 300 <= c.token_count < 400]),
                "400-500": len([c for c in chunks if 400 <= c.token_count < 500]),
                "500+": len([c for c in chunks if c.token_count >= 500]),
            },
        }

        stats_file = output_path / f"stats_{strat_name}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    return results


def load_chunks(input_dir: str = "data/processed", strategy: str = "fixed_size") -> list[Chunk]:
    """Load chunks from disk."""
    input_path = Path(input_dir)
    chunks_file = input_path / f"chunks_{strategy}.jsonl"

    if not chunks_file.exists():
        raise FileNotFoundError(f"No chunks found at {chunks_file}")

    chunks = []
    with open(chunks_file, "r") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))

    return chunks


if __name__ == "__main__":
    # Test chunking
    from src.ingestion.crawler import load_documents

    documents = load_documents("data/raw")
    print(f"Loaded {len(documents)} documents")

    results = chunk_documents(documents, strategy="both")
    for strat, chunks in results.items():
        print(f"{strat}: {len(chunks)} chunks")
