"""Verify fulltext configuration paths and NCBI key are defined."""

from pathlib import Path


def test_fulltext_paths_exist():
    from gralc_rag.config import (
        FULLTEXT_DIR,
        FULLTEXT_RAW_DIR,
        FULLTEXT_PARSED_DIR,
        FULLTEXT_CONDITIONS_DIR,
        FULLTEXT_BENCHMARK_DIR,
        FULLTEXT_RESULTS_DIR,
    )

    assert isinstance(FULLTEXT_DIR, Path)
    assert isinstance(FULLTEXT_RAW_DIR, Path)
    assert isinstance(FULLTEXT_PARSED_DIR, Path)
    assert isinstance(FULLTEXT_CONDITIONS_DIR, Path)
    assert isinstance(FULLTEXT_BENCHMARK_DIR, Path)
    assert isinstance(FULLTEXT_RESULTS_DIR, Path)
    assert "fulltext" in str(FULLTEXT_DIR)


def test_ncbi_api_key_reads_env():
    from gralc_rag.config import NCBI_API_KEY
    assert isinstance(NCBI_API_KEY, str)
