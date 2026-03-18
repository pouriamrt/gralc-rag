#!/bin/bash
# Full pipeline: build corpus, generate QA, evaluate retrieval + generation, generate figures
# Run from project root. Resumes from checkpoints if interrupted.
#
# Usage: bash scripts/run_full_pipeline.sh [max_articles]
#
# Estimated time (CPU, no NCBI API key):
#   2000 articles: ~3-4 hours total
#   5000 articles: ~10-15 hours total

set -e

MAX_ARTICLES=${1:-2000}
echo "=== GraLC-RAG Full Pipeline (max_articles=$MAX_ARTICLES) ==="
echo "Start: $(date)"

# Step 1: Build corpus
echo ""
echo ">>> Step 1/5: Building corpus ($MAX_ARTICLES articles)..."
python scripts/06_build_fulltext_corpus.py --max-articles "$MAX_ARTICLES" --query "biomedical clinical trial research"

# Step 2: Generate QA benchmark (template + LLM if API key exists)
echo ""
echo ">>> Step 2/5: Generating QA benchmark..."
if [ -n "$OPENAI_API_KEY" ]; then
    python scripts/07_generate_crosssection_qa.py --max-articles 100
else
    echo "  (No OPENAI_API_KEY — skipping LLM QA, template only)"
    python scripts/07_generate_crosssection_qa.py --skip-llm
fi

# Step 3: Evaluate retrieval (checkpointed)
echo ""
echo ">>> Step 3/5: Running retrieval evaluation (6 strategies x 3 conditions)..."
python scripts/08_evaluate_fulltext_retrieval.py --conditions intro,partial,fulltext

# Step 4: Generation evaluation (if API key exists)
echo ""
echo ">>> Step 4/5: Running generation evaluation..."
if [ -n "$OPENAI_API_KEY" ]; then
    python scripts/10_evaluate_fulltext_generation.py --n-eval 50 --condition fulltext
else
    echo "  (No OPENAI_API_KEY — skipping generation evaluation)"
fi

# Step 5: Generate figures
echo ""
echo ">>> Step 5/5: Generating figures..."
python scripts/09_generate_fulltext_figures.py

echo ""
echo "=== Pipeline complete! ==="
echo "End: $(date)"
echo ""
echo "Results in: results/fulltext/"
echo "Figures in: figures/"
echo ""
echo "Next: Update paper.tex with new numbers, then run:"
echo "  pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex"
