"""Tests for bulk PMC download function signature and search."""

from unittest.mock import patch, MagicMock
from pathlib import Path


def test_search_pmc_articles_returns_ids():
    from gralc_rag.corpus.downloader import search_pmc_articles

    mock_handle = MagicMock()
    mock_handle.read.return_value = b'<eSearchResult><IdList><Id>12345</Id><Id>67890</Id></IdList><Count>2</Count></eSearchResult>'

    with patch("gralc_rag.corpus.downloader.Entrez.esearch", return_value=mock_handle):
        with patch("gralc_rag.corpus.downloader.Entrez.read", return_value={
            "IdList": ["12345", "67890"],
            "Count": "2",
        }):
            ids = search_pmc_articles(query="cancer", max_results=10)
            assert isinstance(ids, list)
            assert all(isinstance(i, str) for i in ids)


def test_download_pmc_bulk_skips_existing(tmp_path):
    from gralc_rag.corpus.downloader import download_pmc_articles

    (tmp_path / "12345.xml").write_text("<article/>")
    paths = download_pmc_articles(["12345"], save_dir=tmp_path, max_articles=10)
    assert len(paths) == 1
    assert paths[0] == tmp_path / "12345.xml"
