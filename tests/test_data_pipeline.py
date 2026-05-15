"""
Unit tests for data preprocessing pipeline.

Tests for document normalization, sliding window tokenization, and CUAD loading.
"""

import pytest
from pathlib import Path
from src.data.document_normalizer import (
    TextDocumentNormalizer,
    DocumentNormalizerFactory,
    NormalizedDocument,
    DocumentFormat,
)
from src.data.sliding_window import (
    SlidingWindowTokenizer,
    SlidingWindowTokenizerConfig,
    TokenizerStrategy,
    TokenizedSpan,
)
from src.data.cuad_loader import CUADDataset, ClauseCategory


class TestDocumentNormalizer:
    """Tests for DocumentNormalizer."""

    def test_normalize_removes_multiple_newlines(self) -> None:
        """Test that multiple consecutive newlines are collapsed."""
        normalizer = TextDocumentNormalizer()
        
        # Create temp file
        test_content = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        
        result = normalizer._normalize_newlines(test_content)
        
        # Should have at most double newlines
        assert "\n\n\n" not in result
        assert "Line 1" in result and "Line 2" in result and "Line 3" in result

    def test_normalize_removes_page_breaks(self) -> None:
        """Test that page breaks are removed."""
        normalizer = TextDocumentNormalizer()
        
        test_content = "Text\f\nMore text\n[Page 1]\nEven more"
        result = normalizer._remove_page_artifacts(test_content)
        
        assert "\f" not in result
        assert "[Page 1]" not in result
        assert "Text" in result and "More text" in result

    def test_normalize_preserves_indentation(self) -> None:
        """Test that indentation is preserved."""
        normalizer = TextDocumentNormalizer()
        
        test_content = "  Indented line\n    Double indented"
        result = normalizer._normalize_whitespace(test_content)
        
        # Note: normalize_whitespace preserves single indentation
        # but collapses multiple spaces within lines
        assert "Indented line" in result
        assert "Double indented" in result

    def test_estimate_page_count(self) -> None:
        """Test page count estimation."""
        normalizer = TextDocumentNormalizer()
        
        # 3000 chars = ~1 page
        content_1page = "x" * 3000
        assert normalizer._estimate_page_count(content_1page) >= 1
        
        # 9000 chars = ~3 pages
        content_3pages = "x" * 9000
        assert normalizer._estimate_page_count(content_3pages) == 3


class TestSlidingWindowTokenizer:
    """Tests for SlidingWindowTokenizer."""

    def test_tokenize_basic(self) -> None:
        """Test basic tokenization."""
        config = SlidingWindowTokenizerConfig(
            window_size=100,
            overlap_size=20,
        )
        tokenizer = SlidingWindowTokenizer(config)
        
        # Create test content
        content = "x" * 250  # Should create 3 windows
        
        spans = tokenizer.tokenize(content, "test.txt")
        
        assert len(spans) >= 2
        assert all(isinstance(s, TokenizedSpan) for s in spans)
        assert spans[0].window_index == 0
        # Last span should have window_index equal to len(spans)-1
        assert spans[-1].window_index == len(spans) - 1

    def test_tokenize_preserves_total_coverage(self) -> None:
        """Test that tokenization covers entire document."""
        config = SlidingWindowTokenizerConfig(
            window_size=100,
            overlap_size=20,
        )
        tokenizer = SlidingWindowTokenizer(config)
        
        content = "The quick brown fox jumps over the lazy dog. " * 10
        spans = tokenizer.tokenize(content, "test.txt")
        
        # Check coverage (some overlap allowed)
        assert spans[0].char_start == 0
        assert spans[-1].char_end >= len(content) - 1

    def test_tokenize_respects_overlap(self) -> None:
        """Test that overlap is correctly maintained."""
        config = SlidingWindowTokenizerConfig(
            window_size=100,
            overlap_size=30,
        )
        tokenizer = SlidingWindowTokenizer(config)
        
        content = "x" * 250
        spans = tokenizer.tokenize(content, "test.txt")
        
        # Check that consecutive windows overlap
        for i in range(len(spans) - 1):
            assert spans[i].char_end > spans[i + 1].char_start

    def test_tokenize_empty_content_raises(self) -> None:
        """Test that empty content raises ValueError."""
        tokenizer = SlidingWindowTokenizer()
        
        with pytest.raises(ValueError):
            tokenizer.tokenize("", "test.txt")

    def test_tokenize_invalid_config_raises(self) -> None:
        """Test that invalid config raises ValueError."""
        config = SlidingWindowTokenizerConfig(
            window_size=100,
            overlap_size=150,  # Greater than window_size
        )
        tokenizer = SlidingWindowTokenizer(config)
        
        with pytest.raises(ValueError):
            tokenizer.tokenize("test content", "test.txt")

    def test_reconstruct_span_location(self) -> None:
        """Test reconstruction of absolute span location."""
        config = SlidingWindowTokenizerConfig(window_size=50, overlap_size=10)
        tokenizer = SlidingWindowTokenizer(config)
        
        content = "The quick brown fox jumps over the lazy dog"
        spans = tokenizer.tokenize(content, "test.txt")
        
        # Find "brown" in first span
        text_to_find = "brown"
        abs_start, abs_end = tokenizer.reconstruct_span_location(spans, 0, text_to_find)
        
        # Verify it matches the original
        assert content[abs_start:abs_end] == text_to_find

    def test_reconstruct_span_invalid_raises(self) -> None:
        """Test that invalid span raises ValueError."""
        tokenizer = SlidingWindowTokenizer()
        
        config = SlidingWindowTokenizerConfig(window_size=50, overlap_size=10)
        content = "test content"
        spans = tokenizer.tokenize(content, "test.txt")
        
        # Invalid span index
        with pytest.raises(ValueError):
            tokenizer.reconstruct_span_location(spans, 99, "test")
        
        # Text not in span
        with pytest.raises(ValueError):
            tokenizer.reconstruct_span_location(spans, 0, "nonexistent")


class TestCUADDataset:
    """Tests for CUAD dataset loader."""

    def test_clause_categories_defined(self) -> None:
        """Test that all 41 CUAD categories are defined."""
        assert len(CUADDataset.CLAUSE_CATEGORIES) == 41
        
        # Check Q1-Q41
        for i in range(1, 42):
            q_id = f"Q{i}"
            assert any(q == q_id for q in CUADDataset.CLAUSE_CATEGORIES.values())

    def test_sanitize_id(self) -> None:
        """Test ID sanitization."""
        result = CUADDataset._sanitize_id("Test ID / Name")
        
        assert " " not in result
        assert "/" not in result
        assert result.islower()

    def test_sanitize_filename(self) -> None:
        """Test filename sanitization."""
        result = CUADDataset._sanitize_filename("  Test File  ")
        
        assert result == "test_file"

    def test_fuzzy_filename_match(self) -> None:
        """Test fuzzy filename matching."""
        # Similar names should match
        assert CUADDataset._fuzzy_filename_match("agreement_2024", "agreement_2024")
        
        # Partial match
        assert CUADDataset._fuzzy_filename_match("seller_agreement_2024", "seller_agreement")
        
        # Different names should not match
        assert not CUADDataset._fuzzy_filename_match("agreement", "contract")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
