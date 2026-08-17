"""
Sliding window tokenization module.

Implements the SlidingWindowTokenizer for splitting long documents into
overlapping context windows while preserving span coordinates for extractive QA.
Maintains metadata about:
- Document ID
- Page number (if available)
- Character offset within the document
- Window index
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenizerStrategy(Enum):
    """Strategy for tokenization."""

    CHARACTER = "character"  # Split by character count
    SENTENCE = "sentence"  # Split by sentences (future)
    PARAGRAPH = "paragraph"  # Split by paragraphs (future)


@dataclass
class TokenizedSpan:
    """
    Represents a single tokenized window/span from document.

    Attributes:
        window_id: Unique ID for this window (doc_id:window_idx).
        content: Text content of this window.
        doc_filename: Original document filename.
        window_index: Index of this window in document.
        char_start: Character offset where this window starts in original doc.
        char_end: Character offset where this window ends in original doc.
        metadata: Additional metadata (page_num, section_name, etc.).
        overlap_start: Character offset of overlap region start (if window has overlap).
        overlap_end: Character offset of overlap region end (if window has overlap).
    """

    window_id: str
    content: str
    doc_filename: str
    window_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    overlap_start: Optional[int] = None
    overlap_end: Optional[int] = None

    @property
    def length(self) -> int:
        """Length of window content in characters."""
        return len(self.content)

    @property
    def is_last(self) -> bool:
        """Whether this is the last window (heuristic: no overlap_end)."""
        return self.overlap_end is None


@dataclass
class SlidingWindowTokenizerConfig:
    """
    Configuration for sliding window tokenizer.

    Attributes:
        window_size: Size of each window in characters.
        overlap_size: Size of overlap between consecutive windows in characters.
        strategy: Tokenization strategy (character/sentence/paragraph).
        preserve_word_boundaries: Whether to avoid cutting words mid-token.
        min_window_size: Minimum characters for a window (skip if smaller).
    """

    window_size: int = 512  # ~200-300 tokens in typical LLM tokenizers
    overlap_size: int = 128  # ~10-15% overlap for context
    strategy: TokenizerStrategy = TokenizerStrategy.CHARACTER
    preserve_word_boundaries: bool = True
    min_window_size: int = 50


class SlidingWindowTokenizer:
    """
    Tokenizes long documents into overlapping windows while preserving coordinates.

    This is crucial for handling long contracts (50-100+ pages) that exceed LLM
    context windows. Each window maintains:
    - Character offsets for span extraction
    - Original document metadata
    - Overlap information for context preservation
    """

    def __init__(self, config: SlidingWindowTokenizerConfig = None) -> None:
        """
        Initialize tokenizer with configuration.

        Args:
            config: Tokenizer configuration. Uses defaults if None.
        """
        self.config = config or SlidingWindowTokenizerConfig()
        logger.info(
            f"Initialized SlidingWindowTokenizer: "
            f"window_size={self.config.window_size}, "
            f"overlap_size={self.config.overlap_size}"
        )

    def tokenize(
        self,
        content: str,
        doc_filename: str,
        metadata: Dict[str, Any] = None,
    ) -> List[TokenizedSpan]:
        """
        Tokenize document into overlapping windows.

        Args:
            content: Full document text content.
            doc_filename: Original document filename (for traceability).
            metadata: Optional metadata (doc_id, author, etc.).

        Returns:
            List of TokenizedSpan objects representing windows.

        Raises:
            ValueError: If content is empty or configuration is invalid.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        if self.config.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.config.window_size}")

        if self.config.overlap_size >= self.config.window_size:
            raise ValueError(
                f"overlap_size ({self.config.overlap_size}) must be less than "
                f"window_size ({self.config.window_size})"
            )

        logger.info(
            f"Tokenizing document: {doc_filename} "
            f"({len(content)} chars, strategy={self.config.strategy.value})"
        )

        if self.config.strategy == TokenizerStrategy.CHARACTER:
            spans = self._tokenize_by_character(content, doc_filename, metadata)
        else:
            # Future: implement sentence/paragraph strategies
            raise NotImplementedError(f"Strategy {self.config.strategy.value} not yet implemented")

        logger.info(f"✅ Created {len(spans)} windows from {doc_filename}")
        return spans

    def _tokenize_by_character(
        self,
        content: str,
        doc_filename: str,
        metadata: Dict[str, Any] = None,
    ) -> List[TokenizedSpan]:
        """
        Split content into character-based overlapping windows.

        Args:
            content: Full document text.
            doc_filename: Original filename.
            metadata: Optional metadata.

        Returns:
            List of TokenizedSpan objects.
        """
        if metadata is None:
            metadata = {}

        spans: List[TokenizedSpan] = []
        window_index = 0
        step_size = self.config.window_size - self.config.overlap_size

        pos = 0
        while pos < len(content):
            # Calculate window boundaries
            window_start = pos
            window_end = min(pos + self.config.window_size, len(content))

            # Optionally preserve word boundaries
            if (
                self.config.preserve_word_boundaries
                and window_end < len(content)
                and content[window_end] not in (" ", "\n", "\t")
            ):
                # Find the nearest space/newline to the left
                fallback_end = content.rfind(" ", window_start, window_end)
                if fallback_end > window_start + self.config.window_size // 2:
                    window_end = fallback_end + 1

            window_text = content[window_start:window_end].rstrip()

            # Skip very small windows (except last)
            if len(window_text) < self.config.min_window_size and window_end < len(content):
                pos += step_size
                continue

            # Determine overlap region (previous window overlap)
            overlap_start = None
            overlap_end = None
            if window_index > 0 and self.config.overlap_size > 0:
                overlap_start = window_start
                overlap_end = min(window_start + self.config.overlap_size, window_end)

            # Create span
            span = TokenizedSpan(
                window_id=f"{doc_filename}:w{window_index}",
                content=window_text,
                doc_filename=doc_filename,
                window_index=window_index,
                char_start=window_start,
                char_end=window_end,
                metadata=metadata.copy(),
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )

            spans.append(span)
            window_index += 1

            # Move to next window
            pos += step_size

            # Ensure we don't miss the last bit
            if pos >= len(content) and window_end < len(content):
                pos = len(content)

        return spans

    @staticmethod
    def reconstruct_span_location(
        spans: List[TokenizedSpan],
        span_idx: int,
        text_within_window: str,
    ) -> Tuple[int, int]:
        """
        Reconstruct the absolute character position of text found within a window.

        This is useful for extractive QA: if we find an answer span in a window,
        we need to map it back to the original document.

        Args:
            spans: List of all tokenized spans from the document.
            span_idx: Index of the span where text was found.
            text_within_window: Text (exact substring) found within that span.

        Returns:
            Tuple of (absolute_char_start, absolute_char_end) in original document.

        Raises:
            ValueError: If text not found in window or span_idx invalid.
        """
        if span_idx >= len(spans) or span_idx < 0:
            raise ValueError(f"Invalid span index: {span_idx}")

        span = spans[span_idx]

        # Find text within window
        local_start = span.content.find(text_within_window)
        if local_start == -1:
            raise ValueError(f"Text '{text_within_window}' not found in window {span.window_id}")

        local_end = local_start + len(text_within_window)

        # Convert to absolute positions
        abs_start = span.char_start + local_start
        abs_end = span.char_start + local_end

        return (abs_start, abs_end)


class DocumentTokenizationPipeline:
    """
    High-level pipeline that normalizes and tokenizes documents.

    Combines DocumentNormalizer + SlidingWindowTokenizer for a complete pipeline.
    """

    def __init__(
        self,
        tokenizer_config: SlidingWindowTokenizerConfig = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            tokenizer_config: Configuration for sliding window tokenizer.
        """
        self.tokenizer = SlidingWindowTokenizer(tokenizer_config)
        logger.info("Initialized DocumentTokenizationPipeline")

    def process(
        self,
        content: str,
        doc_filename: str,
        doc_metadata: Dict[str, Any] = None,
    ) -> List[TokenizedSpan]:
        """
        Process content through full pipeline (normalize + tokenize).

        Args:
            content: Raw document content (should be pre-normalized).
            doc_filename: Document filename.
            doc_metadata: Optional metadata.

        Returns:
            List of tokenized spans.
        """
        return self.tokenizer.tokenize(content, doc_filename, doc_metadata)
