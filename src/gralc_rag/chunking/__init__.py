"""Chunking strategies for the GraLC-RAG pipeline.

Four strategies are provided:

* **naive** -- fixed-size overlapping token windows.
* **semantic** -- sentence-level splits driven by embedding similarity.
* **late** -- late chunking (Gunther et al. 2024) with document-wide context.
* **structure_aware** -- GraLC-RAG's structure-aware late chunking.
"""

from gralc_rag.chunking.naive import Chunk, naive_chunk
from gralc_rag.chunking.semantic import semantic_chunk
from gralc_rag.chunking.late import late_chunk
from gralc_rag.chunking.structure_aware import (
    compute_boundary_scores,
    structure_aware_chunk,
)

__all__ = [
    "Chunk",
    "naive_chunk",
    "semantic_chunk",
    "late_chunk",
    "compute_boundary_scores",
    "structure_aware_chunk",
]
