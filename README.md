# GraLC-RAG

**Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature**

GraLC-RAG integrates document structure graphs, UMLS knowledge graph infusion, and graph-guided retrieval into the late chunking paradigm for biomedical RAG.

## Key Findings

Standard evaluation metrics like MRR systematically undervalue structural retrieval methods. Our evaluation on 2,359 IMRaD-filtered PubMed Central articles reveals:

- **Precision-breadth trade-off**: Content-similarity methods achieve the highest MRR (0.517) but are structurally blind (SecCov = 1.0). Structure-aware methods retrieve from up to **15.6x more document sections** (SecCov@20 = 15.57).
- **KG infusion bridges the gap**: Ontological enrichment narrows the answer-quality gap to just ΔF1 = 0.009 while maintaining 4.6x section diversity.
- **Multi-section synthesis bottleneck**: Cross-section recall remains 0.000 across all strategies, identifying multi-section reasoning as the critical open problem.

## Architecture

GraLC-RAG operates in five stages:

1. **Document Parsing** — Constructs document structure graphs and UMLS knowledge subgraphs from PubMed Central JATS XML
2. **Full-Document Encoding** — Processes entire documents through a long-context transformer before chunking
3. **Knowledge Graph Infusion** — Injects UMLS ontological signals into token-level representations via GAT attention
4. **Structure-Aware Boundary Detection** — Determines chunk boundaries using structural, semantic, and entity coherence signals
5. **Graph-Guided Retrieval** — Combines dense similarity with KG proximity for hybrid retrieval

## Installation

```bash
# Requires Python 3.11+
uv sync

# Or with pip
pip install -e .
```

## Usage

### Full Pipeline

```bash
# 1. Download PubMedQA corpus + PMC full-text articles
python scripts/01_download_corpus.py

# 2. Index corpus with all chunking strategies
python scripts/02_index_corpus.py

# 3. Evaluate retrieval on PubMedQA
python scripts/03_evaluate_retrieval.py

# 4. Build full-text evaluation corpus (2,359 IMRaD articles)
python scripts/06_build_fulltext_corpus.py

# 5. Generate cross-section QA benchmark (2,033 questions)
python scripts/07_generate_crosssection_qa.py

# 6. Evaluate full-text retrieval with structural coverage metrics
python scripts/08_evaluate_fulltext_retrieval.py

# 7. Evaluate generation quality (requires OpenAI API key)
python scripts/10_evaluate_fulltext_generation.py
```

### Or run everything at once

```bash
bash scripts/run_full_pipeline.sh
```

## Project Structure

```
src/gralc_rag/
├── benchmark/        # Cross-section QA benchmark construction
├── chunking/         # Chunking strategies (naive, semantic, late, structure-aware)
├── corpus/           # PubMed/PMC corpus downloading and parsing
├── evaluation/       # Metrics (MRR, Recall@k, SecCov@k, CS Recall)
├── generation/       # LLM-based answer generation
├── knowledge/        # UMLS entity linking and KG infusion
├── retrieval/        # Dense and graph-guided retrieval
└── config.py         # Configuration
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **MRR** | Ranking accuracy of the single most relevant chunk |
| **Recall@k** | Whether the relevant chunk appears in the top-k |
| **SecCov@k** | Number of distinct document sections in the top-k |
| **CS Recall** | Whether top-k spans multiple required sections |

## Six Retrieval Strategies

1. **Naive** — Fixed-size chunking (256 tokens, 32 overlap)
2. **Semantic** — Embedding similarity-based boundaries
3. **Late Chunking** — Full-document encoding, then sentence-level segmentation
4. **Structure-Aware** — Late chunking with document structure graph boundaries
5. **GraLC-RAG (KG)** — Structure-aware + UMLS knowledge graph infusion
6. **GraLC-RAG (+Graph)** — Full pipeline with graph-guided hybrid retrieval

## Configuration

Set environment variables in `.env`:

```
OPENAI_API_KEY=sk-...          # For generation experiments
NCBI_API_KEY=...               # For PubMed/PMC API access (optional, increases rate limits)
```

## Citation

```bibtex
@article{mortezaagha2026gralcrag,
  title={Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature},
  author={Mortezaagha, Pouria and Rahgozar, Arya},
  year={2026}
}
```

## License

MIT
