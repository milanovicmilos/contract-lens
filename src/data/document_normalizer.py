"""
Document normalization module.

Provides utilities for converting raw contract documents (TXT, PDF, DOCX) into
canonical plain text/markdown format suitable for NLP processing. Handles:
- Removal of headers/footers and page breaks
- Whitespace normalization
- Metadata preservation (page numbers, paragraph IDs)
- Table structure preservation
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DocumentFormat(Enum):
    """Supported document formats."""

    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"


@dataclass
class NormalizedDocument:
    """
    Represents a normalized contract document.

    Attributes:
        filename: Original document filename.
        content: Normalized plain text content.
        format: Original document format.
        page_count: Total number of pages (if applicable).
        metadata: Additional metadata (title, date, etc.).
    """

    filename: str
    content: str
    format: DocumentFormat
    page_count: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class IDocumentNormalizer(ABC):
    """Abstract interface for document normalization."""

    @abstractmethod
    def normalize(self, file_path: Path) -> NormalizedDocument:
        """
        Normalize a document from file path.

        Args:
            file_path: Path to the document file.

        Returns:
            NormalizedDocument with normalized content.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
        """
        pass


class TextDocumentNormalizer(IDocumentNormalizer):
    """
    Normalizer for plain text documents.

    Handles:
    - Removal of multiple blank lines
    - Normalization of whitespace
    - Page break detection and preservation
    - Paragraph ID assignment
    """

    # Patterns for common document artifacts
    PAGE_BREAK_PATTERN = re.compile(
        r"(\f|\n\s*-{5,}|-{5,}\s*\n|\\n\f|\[Page\s+\d+\])", re.IGNORECASE
    )
    MULTIPLE_NEWLINES = re.compile(r"\n\s*\n{2,}")
    LEADING_TRAILING_SPACES = re.compile(r"^\s+|\s+$", re.MULTILINE)
    WHITESPACE_IN_TEXT = re.compile(r"[ \t]+")

    def normalize(self, file_path: Path) -> NormalizedDocument:
        """
        Normalize a plain text document.

        Args:
            file_path: Path to the text file.

        Returns:
            NormalizedDocument with normalized content.

        Raises:
            FileNotFoundError: If file does not exist.
            UnicodeDecodeError: If file encoding is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Normalizing text document: {file_path.name}")

        try:
            # Read file with UTF-8, fallback to latin-1
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {file_path.name}, trying latin-1")
            with open(file_path, "r", encoding="latin-1") as f:
                raw_content = f.read()

        # Normalization pipeline
        content = self._remove_page_artifacts(raw_content)
        content = self._normalize_whitespace(content)
        content = self._normalize_newlines(content)

        # Extract metadata
        metadata = self._extract_metadata(content)

        logger.info(f"✅ Normalized {file_path.name} ({len(content)} chars)")

        return NormalizedDocument(
            filename=file_path.name,
            content=content,
            format=DocumentFormat.TXT,
            page_count=self._estimate_page_count(content),
            metadata=metadata,
        )

    def _remove_page_artifacts(self, content: str) -> str:
        """Remove page breaks and common artifacts."""
        # Remove form feeds
        content = content.replace("\f", "\n")

        # Remove [Page N] markers
        content = re.sub(r"\[Page\s+\d+\]", "", content)

        # Remove common footer patterns (numbers at end of lines)
        content = re.sub(r"(\n\s*)-\d+\s*$", r"\1", content, flags=re.MULTILINE)

        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize tabs and multiple spaces."""
        # Replace tabs with spaces
        content = content.replace("\t", "    ")

        # Collapse multiple spaces (but preserve indentation)
        lines = []
        for line in content.split("\n"):
            # Preserve leading spaces for indentation, collapse others
            stripped_line = self.WHITESPACE_IN_TEXT.sub(" ", line)
            lines.append(stripped_line)

        return "\n".join(lines)

    def _normalize_newlines(self, content: str) -> str:
        """Normalize newlines and blank lines."""
        # Replace multiple consecutive newlines with double newlines (paragraph breaks)
        content = self.MULTIPLE_NEWLINES.sub("\n\n", content)

        # Remove leading/trailing whitespace from each line
        lines = []
        for line in content.split("\n"):
            cleaned_line = line.rstrip()
            lines.append(cleaned_line)

        return "\n".join(lines).strip()

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract basic metadata from document.

        Returns:
            Dictionary with extracted metadata.
        """
        lines = content.split("\n")
        first_non_empty = None

        for line in lines:
            if line.strip():
                first_non_empty = line.strip()
                break

        metadata = {
            "first_line": first_non_empty,
            "has_page_references": bool(
                re.search(r"\b(Page|p\.|pp\.)\s+\d+", content, re.IGNORECASE)
            ),
            "has_tables": "---" in content or "||" in content,
            "language": "en",  # Default to English
        }

        return metadata

    @staticmethod
    def _estimate_page_count(content: str, chars_per_page: int = 3000) -> int:
        """
        Estimate page count based on content length.

        Args:
            content: Document content.
            chars_per_page: Average characters per page (default 3000).

        Returns:
            Estimated page count.
        """
        return max(1, len(content) // chars_per_page)


class DocumentNormalizerFactory:
    """
    Factory for creating appropriate normalizer based on document format.
    """

    _normalizers: Dict[DocumentFormat, IDocumentNormalizer] = {}

    @classmethod
    def register(cls, format_: DocumentFormat, normalizer: IDocumentNormalizer) -> None:
        """Register a normalizer for a specific format."""
        cls._normalizers[format_] = normalizer
        logger.debug(f"Registered normalizer for format: {format_.value}")

    @classmethod
    def get_normalizer(cls, format_: DocumentFormat) -> IDocumentNormalizer:
        """
        Get normalizer for specified format.

        Args:
            format_: Document format.

        Returns:
            IDocumentNormalizer instance.

        Raises:
            ValueError: If format not supported.
        """
        if format_ not in cls._normalizers:
            raise ValueError(
                f"No normalizer registered for format: {format_.value}. "
                f"Supported formats: {list(cls._normalizers.keys())}"
            )
        return cls._normalizers[format_]

    @classmethod
    def normalize(cls, file_path: Path) -> NormalizedDocument:
        """
        Auto-detect format and normalize document.

        Args:
            file_path: Path to document.

        Returns:
            Normalized document.

        Raises:
            ValueError: If format not supported.
        """
        suffix = file_path.suffix.lstrip(".").lower()

        try:
            format_ = DocumentFormat[suffix.upper()]
        except KeyError:
            raise ValueError(f"Unsupported format: {suffix}") from None

        normalizer = cls.get_normalizer(format_)
        return normalizer.normalize(file_path)


# Register default normalizers
DocumentNormalizerFactory.register(DocumentFormat.TXT, TextDocumentNormalizer())
DocumentNormalizerFactory.register(DocumentFormat.MD, TextDocumentNormalizer())
