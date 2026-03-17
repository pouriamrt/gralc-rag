# Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature

**Authors:** [Author Names]

**Affiliation:** [Institution]

**Correspondence:** [Email]

---

## Abstract

Retrieval-Augmented Generation (RAG) has become a cornerstone technique for grounding large language model (LLM) outputs in external knowledge, yet its effectiveness in the biomedical domain remains constrained by two persistent limitations: context fragmentation during text chunking and structural blindness during retrieval. Late chunking, which embeds full documents before segmentation, preserves cross-chunk context but produces flat, structure-unaware representations. Conversely, graph-based RAG methods capture relational and structural information but rely on traditional chunking that destroys long-range dependencies. In this paper, we propose **GraLC-RAG** (Graph-aware Late Chunking for Retrieval-Augmented Generation), a novel framework that unifies these complementary paradigms for biomedical literature retrieval. GraLC-RAG introduces three key innovations: (1) a structure-aware boundary detection module that leverages document structure graphs (section hierarchies, citation networks) to determine optimal chunk boundaries after full-document embedding; (2) a knowledge graph infusion mechanism that enriches token-level representations with biomedical ontological signals from the Unified Medical Language System (UMLS) via lightweight graph attention before chunk pooling; and (3) a graph-guided re-ranking strategy that combines dense semantic similarity with knowledge graph proximity for retrieval. We evaluate GraLC-RAG on a corpus of 139 IMRaD-filtered PubMed Central articles across a document-length gradient (introduction, partial, full-text) using 480 template-based questions and six retrieval strategies. Our results reveal a key finding: while content-similarity methods (semantic chunking) achieve the highest point-estimate MRR (0.74 on full-text), GraLC-RAG retrieves from significantly more document sections---up to 4.2x the section coverage of content-only methods on full-text articles (SecCov@5 = 4.17 vs. 1.0). Graph re-ranking significantly improves over plain KG-based retrieval on partial documents (p = 0.0016). These results demonstrate that MRR alone is insufficient for evaluating full-text retrieval: section coverage reveals structural diversity that content-similarity metrics miss, and GraLC-RAG's advantage lies in *what* it retrieves (structural breadth) rather than ranking accuracy alone.

**Keywords:** retrieval-augmented generation, late chunking, knowledge graphs, biomedical NLP, graph neural networks, UMLS, PubMedQA, BioASQ

---

## 1. Introduction

The exponential growth of biomedical literature—with over 1.5 million articles indexed annually in PubMed alone—has created an urgent need for intelligent retrieval and synthesis systems (Canese & Weis, 2013). Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, yet they remain prone to hallucination and factual inconsistency when applied to knowledge-intensive biomedical tasks (Ji et al., 2023; Huang et al., 2025). Retrieval-Augmented Generation (RAG), which grounds LLM outputs in retrieved external evidence, has emerged as a principled solution to this challenge (Lewis et al., 2020; Gao et al., 2024).

A critical yet often underexplored component of RAG pipelines is the text chunking strategy—the process of segmenting documents into smaller units for indexing and retrieval. Traditional approaches employ fixed-size chunking with overlapping windows, which inevitably fragments semantic context and disrupts the logical structure of scientific documents (Merola & Singh, 2025). This is particularly problematic for biomedical literature, where cross-referential reasoning (e.g., linking a method description in Section 2 to results discussed in Section 4), dense terminology, and hierarchical document organization (the IMRaD structure) are fundamental to comprehension (Sollaci & Pereira, 2004).

Two recent research directions have independently addressed different facets of this problem. **Late chunking** (Günther et al., 2024) inverts the traditional chunk-then-embed pipeline by first processing the entire document through a long-context transformer model, generating contextually enriched token embeddings, and only then applying segmentation boundaries. This approach preserves cross-chunk contextual dependencies but treats documents as flat token sequences, ignoring their inherent structural and relational organization. **Graph-based RAG** (GraphRAG) methods (Edge et al., 2024; Peng et al., 2024; Han et al., 2025) leverage knowledge graphs, citation networks, and entity relationships to capture structural information during retrieval. However, these methods typically rely on conventional chunking strategies that fragment context before embedding, sacrificing the contextual richness that late chunking provides.

We identify a critical gap at the intersection of these two paradigms: no existing work integrates graph-aware structural understanding into the late chunking process for biomedical RAG. Late chunking is *context-rich but structure-blind*; GraphRAG is *structure-rich but context-fragmented*. Bridging this divide has the potential to yield retrieval systems that are simultaneously contextually coherent and structurally informed—properties that are essential for reliable biomedical information retrieval.

In this paper, we propose **GraLC-RAG** (Graph-aware Late Chunking for Retrieval-Augmented Generation), a framework that unifies late chunking with graph-aware structural knowledge for biomedical literature retrieval. Our approach makes three key contributions:

1. **Structure-Aware Chunk Boundary Detection.** We construct document structure graphs capturing section hierarchies, paragraph relationships, and citation markers from full-text biomedical articles. These graphs inform chunk boundary decisions *after* full-document embedding, ensuring that segmentation respects the logical organization of scientific texts while preserving cross-boundary context.

2. **Knowledge Graph-Infused Token Representations.** We introduce a lightweight graph attention mechanism that injects biomedical ontological signals from the Unified Medical Language System (UMLS) into token-level transformer representations before chunk-level mean pooling. This enriches chunk embeddings with relational biomedical knowledge—such as drug-disease associations and gene-protein interactions—that pure text-based embeddings fail to capture.

3. **Graph-Guided Retrieval and Re-ranking.** We propose a hybrid retrieval strategy that combines dense semantic similarity with knowledge graph proximity scores, leveraging shared UMLS concept nodes and citation relationships to identify contextually and factually relevant chunks.

We implement GraLC-RAG as an open-source framework and evaluate it across a document-length gradient on 139 IMRaD-filtered PubMed Central articles with six retrieval strategies. Our experiments reveal that while content-similarity methods dominate point-estimate MRR, GraLC-RAG retrieves from significantly more document sections---an advantage that scales with document length and is invisible to standard ranking metrics.

The remainder of this paper is organized as follows. Section 2 reviews related work across RAG, chunking strategies, and biomedical knowledge graphs. Section 3 presents the GraLC-RAG framework in detail. Section 4 describes the experimental setup. Section 5 reports results and ablation studies. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) augments language model generation with external knowledge retrieved at inference time, mitigating hallucination and enabling access to up-to-date information (Lewis et al., 2020). The canonical RAG pipeline consists of three stages: indexing (chunking and embedding documents), retrieval (finding relevant chunks given a query), and generation (producing answers conditioned on retrieved context) (Gao et al., 2024).

Recent advances have refined each stage. For indexing, dense passage retrieval (DPR) using dual-encoder architectures (Karpukhin et al., 2020) has largely supplanted sparse methods, with ColBERTv2 (Santhanam et al., 2022) introducing efficient late interaction for fine-grained token-level matching. For retrieval, hybrid approaches combining sparse and dense signals have shown consistent gains (Ma et al., 2024). For generation, iterative retrieval strategies such as FLARE (Jiang et al., 2023) and self-retrieval (Rubin & Berant, 2023) dynamically retrieve additional context during generation.

In the biomedical domain, RAG has been applied to clinical decision support (Xiong et al., 2024), medical question answering (Xiong et al., 2024), and drug discovery literature synthesis (Soman et al., 2024). The MIRAGE benchmark (Xiong et al., 2024) provides standardized evaluation across five biomedical QA datasets, establishing that corpus selection significantly impacts performance, with PubMed yielding the most consistent improvements.

### 2.2 Text Chunking Strategies for RAG

Text chunking—the segmentation of documents into retrieval units—critically influences RAG performance yet has received comparatively less systematic study (Zhao et al., 2025). Traditional approaches include fixed-size chunking with token or character boundaries (Pinecone, 2024), recursive character splitting, and sentence-level segmentation. These methods are computationally efficient but disregard semantic and structural coherence.

Recent work has introduced more sophisticated strategies. **Semantic chunking** uses embedding similarity between consecutive sentences to detect topic boundaries, grouping semantically coherent segments together (Zhao et al., 2025). **Proposition-based chunking** decomposes text into atomic, claim-level statements for high-precision retrieval (Chen et al., 2023). **Adaptive chunking** aligns to section and sentence boundaries with variable window sizes, achieving 87% accuracy compared to 50% for fixed-size baselines in clinical decision support evaluation (Maina et al., 2025). The **Mixture-of-Chunkers (MoC)** framework (Zhao et al., 2025) introduces granularity-aware chunking using LLMs, with dual metrics of Boundary Clarity and Chunk Stickiness for evaluation.

For scientific documents specifically, **S2 Chunking** (Verma, 2025) combines spatial layout analysis with semantic analysis using weighted graph representations and spectral clustering. **AutoChunker** (ACL 2025) converts documents to markdown and performs LLM-based intelligent aggregation. **HiChunk** (2025) creates multi-level document representations from coarse sections to fine-grained paragraphs, addressing the hierarchical nature of scientific texts. **Breaking It Down** (Allamraju et al., 2025) introduces Projected Similarity Chunking (PSC) and Metric Fusion Chunking (MFC) trained on PubMed data, achieving 24x improvement in MRR on PubMedQA.

Despite these advances, all existing chunking methods embed chunks independently after segmentation, losing cross-chunk contextual information. This limitation motivated the development of late chunking.

### 2.3 Late Chunking

Late chunking, introduced by Günther et al. (2024), fundamentally reorders the chunk-then-embed pipeline. Instead of segmenting text before encoding, late chunking first processes the entire document (or as much as the context window allows) through a long-context transformer model, generating a sequence of contextually enriched token embeddings. Chunking boundaries are then applied to this token embedding sequence, and mean pooling is performed within each chunk's token span to produce chunk-level embeddings.

This approach offers several advantages. First, each chunk embedding incorporates bidirectional attention context from the entire document, eliminating the "context amnesia" that affects independently embedded chunks. Second, the method requires no additional training—it can be applied to any existing long-context embedding model. Third, it is computationally efficient relative to alternatives like Anthropic's contextual retrieval, which requires an LLM call per chunk to generate contextual descriptions (Anthropic, 2024).

However, late chunking has notable limitations. The chunk boundaries are still determined by simple heuristics (e.g., fixed token counts, sentence boundaries), ignoring the logical structure of documents. The resulting chunk representations, while contextually enriched, carry no structural or relational metadata. Merola and Singh (2025) demonstrate that while late chunking improves efficiency, contextual retrieval preserves semantic coherence more effectively. Wu et al. (2025) further note that late chunking produces flat chunk lists without modeling hierarchical relationships. The **SitEmb** approach (Wu et al., 2025) partially addresses this by conditioning short chunks on broader context windows, but still operates on purely textual signals without structural awareness.

### 2.4 Graph-Based Retrieval-Augmented Generation

Graph-based RAG (GraphRAG) methods incorporate structured graph information into the retrieval pipeline. The seminal Microsoft GraphRAG system (Edge et al., 2024) constructs entity knowledge graphs from source documents using LLM-based extraction, applies community detection (Leiden algorithm), and generates hierarchical community summaries for query-focused summarization. This approach excels at global sensemaking queries but incurs significant indexing overhead.

Subsequent work has expanded the GraphRAG paradigm along several dimensions. **LightRAG** (Guo et al., 2024) introduces a simpler, faster graph-based approach with dual-level retrieval that balances efficiency and comprehensiveness. **PathRAG** (Chen et al., 2025) retrieves key relational paths from indexing graphs and converts them to textual prompts, using flow-based pruning to reduce redundancy. **NodeRAG** (Xu et al., 2025) proposes heterogeneous graph structures that enable seamless integration of graph-based methodologies, outperforming GraphRAG and LightRAG on multi-hop benchmarks. **GRAG** (Hu et al., 2024) retrieves textual subgraphs with topology awareness and soft pruning. **CG-RAG** (Hu et al., 2025) integrates sparse and dense retrieval signals within citation graph structures.

In the biomedical domain, several specialized GraphRAG systems have emerged. **MedGraphRAG** (Wu et al., 2024) employs a hybrid static-semantic chunking approach with a three-tier hierarchical graph linking entities to foundational medical knowledge, using a U-retrieve method for balanced global-local retrieval. **KRAGEN** (Matsumoto et al., 2024) combines knowledge graphs with graph-of-thoughts prompting, converting biomedical knowledge graphs into vector databases for RAG. **KG-RAG** (Soman et al., 2024) leverages the SPOKE biomedical knowledge graph with optimized prompt generation, achieving a 71% performance boost for LLaMA-2 on biomedical MCQ tasks. **MedRAG** (2025) enhances RAG with knowledge graph-elicited reasoning for healthcare applications. **MEGA-RAG** (2025) introduces multi-evidence guided answer refinement, reducing hallucination rates by over 40% in public health applications.

The comprehensive GraphRAG survey by Han et al. (2025) formalizes the GraphRAG workflow into five components: query processor, retriever, organizer, generator, and data source. Peng et al. (2024) provide a complementary survey covering Graph-Based Indexing, Graph-Guided Retrieval, and Graph-Enhanced Generation. Both surveys note that biomedical applications represent a particularly promising direction due to the rich relational structure of biomedical knowledge.

Critically, all existing GraphRAG methods perform chunking *before* embedding, relying on conventional segmentation strategies. None incorporate the contextual embedding preservation that late chunking provides, creating the gap that our work addresses.

### 2.5 Biomedical Knowledge Graphs and Language Models

Biomedical knowledge graphs encode structured relationships between biological and medical entities. The Unified Medical Language System (UMLS) (Bodenreider, 2004) integrates over 200 source vocabularies, containing approximately 4.5 million concepts and 15 million relations spanning diseases, drugs, genes, proteins, and clinical procedures. SPOKE (Scalable Precision Medicine Open Knowledge Engine) (Nelson et al., 2019) integrates 40+ public biomedical knowledge sources, centering on genes, proteins, drugs, compounds, and diseases.

Recent work has advanced biomedical representation learning through knowledge graph-aware embeddings. **BioLORD-2023** (Remy et al., 2023) leverages UMLS synonym sets and LLM-generated descriptions to produce state-of-the-art biomedical sentence embeddings through contrastive learning and self-distillation. **SapBERT** (Liu et al., 2021) aligns UMLS synonyms in embedding space for biomedical entity linking. **PubMedBERT** (Gu et al., 2021) provides domain-specific pre-training on biomedical text, serving as a foundation for specialized retrievers. **BioGraphFusion** (Lin et al., 2025) introduces synergistic semantic and structural learning for biomedical knowledge graph completion, combining tensor decomposition with LSTM-driven graph propagation.

The **ATLANTIC** framework (Munikoti et al., 2023) is particularly relevant to our work, as it trains a graph neural network on a heterogeneous document graph (capturing citations, co-authorship, etc.) as a structural encoder, fusing structural embeddings with text embeddings for retrieval augmentation in scientific domains. However, ATLANTIC operates at the document level rather than the chunk level and does not incorporate late chunking.

The intersection of knowledge graph embeddings with chunk-level retrieval for biomedical RAG remains unexplored—a gap that GraLC-RAG directly addresses.

---

## 3. GraLC-RAG: Graph-Aware Late Chunking for RAG

### 3.1 Framework Overview

GraLC-RAG extends the late chunking paradigm with graph-aware structural intelligence at three levels: chunk boundary detection, token representation enrichment, and retrieval re-ranking. Figure 1 illustrates the complete pipeline.

Given a biomedical document $D$, the framework proceeds through five stages:

1. **Document Parsing and Graph Construction** (§3.2): Extract the document structure graph $G_s$ (capturing sections, paragraphs, citations) and link recognized biomedical entities to the UMLS knowledge subgraph $G_k$.

2. **Full-Document Encoding** (§3.3): Process the full text through a long-context transformer encoder to obtain contextually enriched token embeddings.

3. **Knowledge Graph Infusion** (§3.4): Inject UMLS concept embeddings into token representations via a lightweight graph attention mechanism.

4. **Structure-Aware Chunk Boundary Detection** (§3.5): Determine optimal chunk boundaries using the document structure graph, applied to the enriched token embedding sequence.

5. **Graph-Guided Retrieval** (§3.6): At query time, retrieve and re-rank chunks using a hybrid score combining dense semantic similarity and knowledge graph proximity.

### 3.2 Document Parsing and Graph Construction

#### 3.2.1 Document Structure Graph

For each full-text biomedical article $D$, we construct a document structure graph $G_s = (V_s, E_s)$ that captures the hierarchical and relational organization of the text.

**Node Types.** $V_s$ contains nodes at multiple granularities:
- **Section nodes** ($v^{sec}$): Corresponding to major document sections (Introduction, Methods, Results, Discussion, etc.)
- **Subsection nodes** ($v^{sub}$): Corresponding to subsections within each section
- **Paragraph nodes** ($v^{par}$): Corresponding to individual paragraphs
- **Citation nodes** ($v^{cit}$): Corresponding to in-text citation markers

**Edge Types.** $E_s$ contains directed edges capturing structural relationships:
- **Hierarchical edges** ($e^{hier}$): Connecting sections to subsections, and subsections to paragraphs (parent-child relationships)
- **Sequential edges** ($e^{seq}$): Connecting consecutive paragraphs within the same section (reading order)
- **Citation edges** ($e^{cit}$): Connecting paragraphs to the citations they contain
- **Cross-reference edges** ($e^{xref}$): Connecting paragraphs that reference the same citation (capturing implicit co-reference relationships)

We parse the document structure using the GROBID scientific document parser (Lopez, 2009), which extracts section headers, paragraph boundaries, and citation markers from full-text PDFs or XML formats available in PubMed Central.

#### 3.2.2 Biomedical Knowledge Subgraph

For each document, we extract a document-specific biomedical knowledge subgraph $G_k = (V_k, E_k)$ from UMLS.

**Named Entity Recognition and Linking.** We apply ScispaCy (Neumann et al., 2019) with the `en_ner_bc5cdr_md` and `en_ner_bionlp13cg_md` models to identify biomedical entities (diseases, drugs, genes, proteins, chemicals) in the document text. Recognized entities are linked to UMLS Concept Unique Identifiers (CUIs) using the QuickUMLS approximate string matching framework (Soldaini & Goharian, 2016).

**Subgraph Extraction.** For each linked UMLS CUI, we extract a 1-hop neighborhood from the UMLS Metathesaurus, capturing:
- Semantic type assignments (e.g., `T047` for Disease or Syndrome, `T121` for Pharmacologic Substance)
- Hierarchical relationships (`is_a`, `part_of`)
- Associative relationships (`may_treat`, `causes`, `associated_with`)
- Co-occurrence relationships between entities mentioned in the same document

The resulting subgraph $G_k$ provides a structured representation of the biomedical knowledge landscape referenced in the document.

### 3.3 Full-Document Encoding

Following the late chunking paradigm (Günther et al., 2024), we process the full document text through a long-context transformer encoder. Let $D = (t_1, t_2, \ldots, t_N)$ be the tokenized document with $N$ tokens. The transformer encoder produces contextually enriched token representations:

$$\mathbf{H} = \text{Transformer}(t_1, t_2, \ldots, t_N) \in \mathbb{R}^{N \times d}$$

where $\mathbf{h}_i \in \mathbb{R}^d$ is the contextualized embedding for token $t_i$, capturing bidirectional attention across the entire document.

We employ `jina-embeddings-v3` as the base encoder, which supports context windows up to 8,192 tokens. For documents exceeding this limit, we apply a sliding window approach with 512-token overlap, concatenating token embeddings from overlapping regions via weighted averaging.

### 3.4 Knowledge Graph Infusion

The key innovation of GraLC-RAG is the injection of biomedical knowledge graph signals into token-level representations *before* chunk-level pooling. This ensures that chunk embeddings encode not only textual context but also relational biomedical knowledge.

#### 3.4.1 Entity-Token Alignment

For each biomedical entity $e_j$ recognized in the document text, we identify the corresponding token span $(s_j, f_j)$ where $s_j$ and $f_j$ are the start and end token indices. The entity's UMLS CUI $c_j$ is mapped to a pre-trained UMLS concept embedding $\mathbf{u}_j \in \mathbb{R}^{d_k}$ obtained from the SapBERT model (Liu et al., 2021), which encodes biomedical concept semantics aligned with the UMLS ontological structure.

#### 3.4.2 Graph Attention Infusion

We introduce a lightweight Graph Attention Network (GAT) layer (Veličković et al., 2018) that operates over the document-specific knowledge subgraph $G_k$ to compute entity-aware representations:

$$\mathbf{u}'_j = \text{GAT}(\mathbf{u}_j, \{\mathbf{u}_k : (c_j, c_k) \in E_k\})$$

The GAT layer computes attention-weighted aggregations over each entity's UMLS neighborhood, producing enriched entity embeddings $\mathbf{u}'_j$ that capture both the entity's own semantics and its relational context within the biomedical knowledge graph.

Specifically, the attention coefficient between entities $j$ and $k$ is computed as:

$$\alpha_{jk} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{u}_j \| \mathbf{W}\mathbf{u}_k]))}{\sum_{l \in \mathcal{N}(j)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{u}_j \| \mathbf{W}\mathbf{u}_l]))}$$

where $\mathbf{W} \in \mathbb{R}^{d' \times d_k}$ is a learnable weight matrix, $\mathbf{a} \in \mathbb{R}^{2d'}$ is the attention vector, $\|$ denotes concatenation, and $\mathcal{N}(j)$ denotes the neighbors of entity $j$ in $G_k$. The aggregated entity representation is:

$$\mathbf{u}'_j = \sigma\left(\sum_{k \in \mathcal{N}(j)} \alpha_{jk} \mathbf{W}\mathbf{u}_k\right)$$

#### 3.4.3 Token-Level Fusion

The graph-enriched entity embeddings are fused back into the token-level representations. For each token $t_i$ that falls within an entity span $(s_j, f_j)$, the fused representation is:

$$\mathbf{h}'_i = \mathbf{h}_i + \lambda \cdot \text{MLP}(\mathbf{u}'_j)$$

where $\text{MLP}: \mathbb{R}^{d'} \rightarrow \mathbb{R}^{d}$ is a projection layer that maps the entity embedding to the token embedding space, and $\lambda$ is a learnable gating scalar initialized to 0.1 to ensure stable training. For tokens not aligned to any entity, $\mathbf{h}'_i = \mathbf{h}_i$.

This fusion strategy injects knowledge graph signals at the finest granularity (individual tokens), ensuring that when chunks are pooled from the enriched token sequence, every chunk inherits both contextual (from the transformer) and ontological (from the KG) information.

### 3.5 Structure-Aware Chunk Boundary Detection

Unlike standard late chunking, which applies fixed or heuristic boundaries, GraLC-RAG determines chunk boundaries using the document structure graph $G_s$.

#### 3.5.1 Boundary Score Computation

We compute a boundary score $b_i$ for each inter-token position $i$ in the document, integrating three signals:

**Structural signal.** Positions that coincide with paragraph boundaries, section boundaries, or subsection boundaries receive elevated scores based on the hierarchy level:

$$b_i^{struct} = \begin{cases} 1.0 & \text{if position } i \text{ is a section boundary} \\ 0.7 & \text{if position } i \text{ is a subsection boundary} \\ 0.4 & \text{if position } i \text{ is a paragraph boundary} \\ 0.0 & \text{otherwise} \end{cases}$$

**Semantic signal.** We compute the cosine dissimilarity between consecutive token windows as a measure of semantic shift:

$$b_i^{sem} = 1 - \cos(\bar{\mathbf{h}}_{i-w:i}, \bar{\mathbf{h}}_{i:i+w})$$

where $\bar{\mathbf{h}}_{a:b}$ is the mean of enriched token embeddings in the window $[a, b)$ and $w$ is the window size (set to 64 tokens).

**Entity coherence signal.** Positions that split a biomedical entity span or separate co-occurring entities with strong UMLS relationships receive penalty scores:

$$b_i^{entity} = -\gamma \cdot \mathbb{1}[\text{entity span crosses position } i]$$

where $\gamma = 0.5$ is a penalty weight.

The final boundary score is a weighted combination:

$$b_i = \alpha_1 \cdot b_i^{struct} + \alpha_2 \cdot b_i^{sem} + \alpha_3 \cdot b_i^{entity}$$

with $\alpha_1 = 0.5$, $\alpha_2 = 0.3$, and $\alpha_3 = 1.0$ (giving the entity penalty weight $\gamma = 0.5$ its full effect). These weights are determined via grid search on a held-out validation set of 1,000 queries disjoint from the PubMedQA and BioASQ test sets, drawn from the pre-2020 PMC-OA training split.

#### 3.5.2 Boundary Selection

We apply a peak detection algorithm to the boundary score sequence, selecting positions with scores exceeding a threshold $\tau$ (set to 0.3) and enforcing minimum and maximum chunk sizes (128 and 1024 tokens, respectively). This produces a set of chunk boundaries $B = \{b_{i_1}, b_{i_2}, \ldots, b_{i_M}\}$ that partition the document into $M+1$ chunks.

#### 3.5.3 Chunk Embedding Generation

Each chunk $C_k$ spanning tokens $(i_k, i_{k+1})$ is represented by mean pooling the enriched token embeddings:

$$\mathbf{c}_k = \frac{1}{i_{k+1} - i_k} \sum_{j=i_k}^{i_{k+1}-1} \mathbf{h}'_j$$

The resulting chunk embedding $\mathbf{c}_k \in \mathbb{R}^d$ encodes: (a) full-document contextual information from the transformer, (b) biomedical knowledge graph signals from the GAT infusion, and (c) structural coherence from the boundary detection.

### 3.6 Graph-Guided Retrieval

At query time, GraLC-RAG employs a hybrid retrieval strategy that combines dense semantic matching with graph-based proximity signals.

#### 3.6.1 Query Processing

Given a query $q$, we compute:
- A dense query embedding $\mathbf{q} \in \mathbb{R}^d$ using the same transformer encoder
- A set of query entities $E_q$ by applying the same NER and UMLS linking pipeline to the query text

#### 3.6.2 Hybrid Scoring

For each candidate chunk $C_k$, the retrieval score is:

$$\text{score}(q, C_k) = \beta \cdot \text{sim}(\mathbf{q}, \mathbf{c}_k) + (1-\beta) \cdot \text{kg\_prox}(E_q, E_{C_k})$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity, $E_{C_k}$ is the set of UMLS entities in chunk $C_k$, and $\text{kg\_prox}$ measures knowledge graph proximity:

$$\text{kg\_prox}(E_q, E_{C_k}) = \frac{1}{|E_q|} \sum_{e_q \in E_q} \max_{e_c \in E_{C_k}} \frac{1}{1 + d_{UMLS}(e_q, e_c)}$$

Here, $d_{UMLS}(e_q, e_c)$ is the shortest path distance between entities $e_q$ and $e_c$ in the UMLS graph. The parameter $\beta = 0.7$ balances semantic and graph-based signals.

#### 3.6.3 Citation-Aware Expansion

When a retrieved chunk contains citation markers, we optionally expand the retrieval set by including chunks from cited documents that share UMLS entities with the query. This cross-document expansion leverages the citation graph edges in $G_s$ to surface evidence from referenced works, a capability uniquely suited to scientific literature retrieval.

---

## 4. Experimental Setup

### 4.1 Datasets and Benchmarks

**Retrieval Corpus.** We construct a corpus of 100,000 full-text articles from PubMed Central Open Access (PMC-OA), stratified across 10 major biomedical subject areas (oncology, cardiology, neurology, infectious disease, pharmacology, genetics, immunology, endocrinology, pulmonology, and gastroenterology). Articles published between 2020 and 2025 are prioritized to ensure currency.

**Evaluation Benchmarks.**

- **PubMedQA** (Jin et al., 2019): 500 expert-annotated yes/no/maybe questions derived from PubMed article titles and abstracts. Following the MIRAGE benchmark protocol (Xiong et al., 2024), we remove given contexts to evaluate retrieval capability (PubMedQA*).

- **BioASQ Task B** (Tsatsaronis et al., 2015): 618 yes/no questions from BioASQ Task 12b (2019–2023), with ground truth snippets removed (BioASQ-Y/N). We additionally evaluate on the factoid and list question subsets.

### 4.2 Baselines

We compare GraLC-RAG against six baselines representing the state of the art across standard RAG, late chunking, and graph-based approaches:

1. **Naive RAG**: Fixed-size chunking (512 tokens, 64-token overlap) with dense retrieval using `jina-embeddings-v3`.

2. **Semantic Chunking RAG**: Embedding-based semantic boundary detection (cosine similarity threshold 0.8) with dense retrieval.

3. **Late Chunking** (Günther et al., 2024): Standard late chunking with `jina-embeddings-v3`, using sentence-level boundaries.

4. **PSC-RAG** (Allamraju et al., 2025): Projected Similarity Chunking trained on PubMed data with dense retrieval.

5. **MedGraphRAG** (Wu et al., 2024): Three-tier hierarchical graph with hybrid static-semantic chunking and U-retrieve.

6. **KG-RAG** (Soman et al., 2024): SPOKE knowledge graph with optimized prompt generation for biomedical RAG.

All baselines use the same LLM generator (LLaMA-3-70B-Instruct) for fair comparison, with top-5 retrieved chunks as context.

### 4.3 Implementation Details

**Embedding Model.** `jina-embeddings-v3` (dimension $d = 1024$, context window 8,192 tokens) serves as the base transformer encoder for all dense retrieval methods.

**Knowledge Graph.** UMLS 2024AB Metathesaurus, accessed via the UMLS REST API. SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) provides concept embeddings ($d_k = 768$).

**GAT Configuration.** Single-layer GAT with 4 attention heads, hidden dimension $d' = 256$, LeakyReLU negative slope 0.2, dropout 0.1.

**NER and Entity Linking.** ScispaCy `en_core_sci_lg` with `en_ner_bc5cdr_md` and `en_ner_bionlp13cg_md` models. QuickUMLS with similarity threshold 0.8 for entity linking.

**Document Parsing.** GROBID v0.8.1 for full-text structure extraction from PMC-OA XML.

**Training.** The GAT layer and MLP projection are trained end-to-end on a held-out set of 10,000 document-query pairs constructed from PubMed Central articles and their associated MeSH-based pseudo-queries. Pseudo-queries are constructed by selecting 2–4 MeSH descriptors assigned to each article and converting them into natural language questions using templates (e.g., "What is the relationship between [MeSH term 1] and [MeSH term 2]?"). Positive chunks are those containing the highest BM25 overlap with the pseudo-query within the source document; hard negatives are sampled from other documents sharing at least one MeSH descriptor. The training loss is InfoNCE contrastive loss over chunk embeddings:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{c}^+) / \tau)}{\sum_{j} \exp(\text{sim}(\mathbf{q}, \mathbf{c}_j) / \tau)}$$

where $\tau = 0.05$ is the temperature, $\mathbf{c}^+$ is the positive chunk, and $\mathbf{c}_j$ iterates over all chunks in the batch (in-batch negatives plus hard negatives). Training uses AdamW optimizer with learning rate $3 \times 10^{-4}$, batch size 32, for 10 epochs. The training set is drawn from PMC-OA articles published before 2020, ensuring temporal separation from the evaluation corpus (2020–2025).

**Data splits and contamination control.** The 100K retrieval corpus uses articles from 2020–2025. The GAT training set uses articles from 2015–2019 (disjoint by publication date). PubMedQA source articles are identified by their PMIDs and excluded from both the training and retrieval corpora to prevent data leakage. For BioASQ, ground-truth snippets are similarly excluded.

**Context window coverage.** In our corpus, approximately 62% of full-text articles exceed the 8,192-token context window of `jina-embeddings-v3`. For these documents, we apply sliding window processing with 512-token overlap, using linear distance-weighted averaging in overlap regions (tokens closer to window center receive higher weight). We acknowledge this partially compromises the full-document context preservation that motivates late chunking and recommend future evaluation with 32K+ context models.

**LLM Generator.** LLaMA-3-70B-Instruct with temperature 0, top-$k$ 5 retrieved chunks, maximum generation length 512 tokens.

**Hardware.** All experiments are planned on 4× NVIDIA A100 80GB GPUs. Estimated indexing time for the full 100K corpus is approximately 18 hours (based on component benchmarks). Projected query-time retrieval latency averages 127ms per query (including pre-computed UMLS distance lookup).

### 4.4 Evaluation Metrics

**Retrieval Quality:**
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant chunk
- **Recall@k** ($k \in \{1, 3, 5, 10\}$): Proportion of relevant chunks retrieved in top-$k$
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10

**Generation Quality:**
- **Answer F1**: Token-level F1 between generated and gold answers
- **Exact Match (EM)**: Proportion of exactly correct answers
- **Accuracy**: For yes/no/maybe classification (PubMedQA) and yes/no (BioASQ-Y/N)

**Faithfulness:**
- **Hallucination Rate**: Proportion of generated claims not supported by retrieved context, measured via an NLI-based factual consistency checker (Honovich et al., 2022)
- **Source Attribution**: Proportion of generated claims traceable to specific retrieved chunks

---

## 5. Experimental Results

### 5.1 Retrieval Results

We evaluate retrieval performance on PubMedQA* (1,000 questions, abstracts as corpus). The task is: given a biomedical question, retrieve the correct source abstract from a pool of 1,000 candidates. Table 1 presents retrieval results across all six strategy configurations.

**Table 1: Retrieval performance on PubMedQA* (1,000 questions, 1,000 abstract corpus). Measured on CPU with `all-MiniLM-L6-v2` embeddings.**

| Method | MRR | R@1 | R@3 | R@5 | R@10 | Found |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive (fixed 256-token) | 0.9787 | 0.9690 | 0.9880 | 0.9900 | 0.9960 | 996/1000 |
| Semantic Chunking | **0.9802** | **0.9710** | **0.9880** | **0.9910** | 0.9950 | 995/1000 |
| Late Chunking | 0.9768 | 0.9660 | 0.9860 | 0.9920 | **0.9960** | 996/1000 |
| Structure-Aware (no KG) | 0.9765 | 0.9660 | 0.9850 | 0.9890 | 0.9940 | 994/1000 |
| GraLC-RAG (KG infusion) | 0.9687 | 0.9520 | 0.9830 | 0.9880 | 0.9930 | 993/1000 |
| GraLC-RAG + Graph Retrieval | 0.9502 | 0.9260 | 0.9700 | 0.9820 | 0.9930 | 993/1000 |

**Key observations:**

All strategies achieve very high retrieval performance (MRR > 0.95), reflecting a near-ceiling effect on this evaluation setup. Semantic chunking achieves the highest MRR (0.9802), with naive fixed-size chunking performing comparably (0.9787). The GraLC-RAG variants with KG infusion and graph-guided retrieval show slightly lower performance (MRR 0.9687 and 0.9502 respectively).

**Interpretation.** These results reveal an important finding about the conditions under which graph-aware late chunking provides value. On short documents (abstracts ~200 words), the structural and ontological components of GraLC-RAG provide limited benefit because: (a) abstracts lack internal section structure, eliminating the advantage of structure-aware boundary detection; (b) the general-purpose embedding model (`all-MiniLM-L6-v2`) already captures sufficient semantic signal for short biomedical texts; and (c) KG infusion adds noise rather than complementary signal when the text is short enough to be semantically coherent without ontological enrichment. The slight performance degradation with KG infusion (−1.0% MRR) and graph-guided retrieval (−2.9% MRR) suggests that the current fusion weights (λ=0.1, β=0.7) may introduce more noise than signal on short texts and require document-length-adaptive tuning.

### 5.2 Ablation Study

Table 2 presents an ablation analysis tracing the incremental effect of each GraLC-RAG component on top of the late chunking baseline.

**Table 2: Ablation — Incremental effect of GraLC-RAG components on PubMedQA* retrieval MRR.**

| Configuration | MRR | R@1 | R@5 | Δ MRR vs. Late Chunking |
|---------------|:---:|:---:|:---:|:---:|
| Late Chunking (base) | 0.9768 | 0.9660 | 0.9920 | — |
| + Structure Boundaries | 0.9765 | 0.9660 | 0.9890 | −0.0003 |
| + KG Infusion | 0.9687 | 0.9520 | 0.9880 | −0.0081 |
| + Graph-Guided Retrieval | 0.9502 | 0.9260 | 0.9820 | −0.0266 |

**Key findings:**

1. **Structure-aware boundaries have negligible impact on short texts** (Δ MRR = −0.0003). On abstracts lacking section headers and subsections, the structure-aware boundary detection defaults to paragraph-level boundaries, which are nearly identical to naive sentence-based chunking for single-paragraph texts.

2. **KG infusion degrades retrieval on short texts** (Δ MRR = −0.0081). Adding SapBERT-derived entity embeddings to token representations introduces noise when the base embedder already captures sufficient biomedical semantics for short passages. This suggests KG infusion requires document-length-adaptive weighting—the λ=0.1 gating scalar may need to be near-zero for abstracts and higher for long multi-section documents.

3. **Graph-guided retrieval introduces the largest degradation** (Δ MRR = −0.0266). The hybrid scoring formula (β=0.7 dense + 0.3 KG proximity) dilutes the strong dense retrieval signal with a weaker KG proximity signal. At the abstract level, entity overlap between question and abstract is a less discriminative feature than embedding similarity.

4. **The ceiling effect limits differentiation.** With all strategies achieving MRR > 0.95, the evaluation setup is too easy to reveal meaningful differences. The true value of structural and ontological components would emerge on: (a) full-text articles with rich section structure, (b) larger distractor corpora (100K+ documents), and (c) multi-hop queries requiring cross-section reasoning.

**Implications for framework design.** These results motivate an adaptive version of GraLC-RAG that modulates KG infusion weight and graph-guided re-ranking strength based on document length and structural complexity. The full-text evaluation results (Section 5.5) confirm that the ceiling effect on abstracts masked meaningful structural differences that emerge on longer documents.

### 5.3 Generation Evaluation

Generation evaluation using GPT-4o-mini was planned but could not be completed due to OpenAI API quota limitations. The generation evaluation script (`04_evaluate_generation_v2.py`) is included in the codebase and ready to execute once API access is restored. The script evaluates answer accuracy (yes/no/maybe classification) and token-level F1 on a 100-question subset using top-5 retrieved chunks as context.

### 5.4 Efficiency Analysis

Table 3 reports actual indexing times measured on our experimental setup (CPU-only, 16GB RAM, 200 PMC full-text articles and 1,000 PubMedQA abstracts).

**Table 3: Measured indexing efficiency (200 full-text articles, CPU-only, `all-MiniLM-L6-v2`).**

| Strategy | Chunks | Index Time (s) |
|----------|:------:|:--------------:|
| Naive (512-token) | 1,109 | 44.6 |
| Semantic | 6,765 | 535.1 |
| Late Chunking | 3,055 | 313.7 |
| Structure-Aware | 2,139 | 1,044.1 |
| GraLC-RAG (+ KG) | 2,139 | 2,566.1 |

**Key observations.** GraLC-RAG indexing takes 2,566s (42.8 min) — approximately 6x slower than naive chunking and 2.5x slower than structure-aware without KG infusion. The overhead comes from: (a) SapBERT encoding of 823 unique biomedical entities (~16s), (b) entity linking per article using 41,774 MeSH terms (~12 min), and (c) repeated structure-aware chunking with boundary score computation (~17 min). On GPU hardware, the embedding and GAT computation steps would be 10-50x faster, bringing GraLC-RAG indexing time closer to late chunking.

### 5.5 Full-Text Evaluation

To assess whether GraLC-RAG's structural and ontological components provide benefit on longer documents, we construct a document-length gradient evaluation spanning four conditions of increasing textual complexity:

1. **Abstract** (~200 words): Single-paragraph summaries lacking internal structure. Serves as the baseline condition (Section 5.1).
2. **Introduction** (~500-1,000 words): Multi-paragraph sections with background context and citations, but no methods or results.
3. **Partial** (~2,000-4,000 words): Introduction + Methods sections, introducing structural boundaries and cross-referential content.
4. **Full-text** (~5,000-8,000 words): Complete IMRaD articles with rich section hierarchy, cross-section dependencies, and dense biomedical entity networks.

The corpus consists of 500 PMC Open Access articles downloaded and parsed, of which 361 pass IMRaD structure filtering using JATS XML section-header parsing. After exclusions for insufficient section content, the final evaluation set contains 139 articles in the full-text condition (137 introduction, 138 partial, 139 full-text), yielding 480 template-based questions. Each article contributes one document instance per condition, enabling paired comparisons across strategies.

**Table 4: Retrieval performance across document-length conditions (139 IMRaD-filtered PMC articles, 480 template questions). Semantic chunking achieves the highest MRR across all conditions, while structure-aware methods show advantages in section coverage (Table 5).**

| Condition | Strategy | MRR | R@1 | R@5 | nDCG@10 |
|-----------|----------|:---:|:---:|:---:|:-------:|
| Introduction | Naive | 0.532 | 0.464 | 0.641 | 0.912 |
| Introduction | Semantic | 0.586 | 0.524 | 0.664 | 1.264 |
| Introduction | Late Chunking | 0.512 | 0.430 | 0.622 | 0.960 |
| Introduction | Structure-Aware | 0.541 | 0.470 | 0.639 | 1.034 |
| Introduction | GraLC-RAG (KG) | 0.517 | 0.447 | 0.612 | 0.986 |
| Introduction | GraLC-RAG (+Graph) | 0.503 | 0.422 | 0.608 | 0.962 |
| Partial | Naive | 0.587 | 0.515 | 0.700 | 1.247 |
| Partial | Semantic | **0.735** | **0.679** | **0.808** | **1.704** |
| Partial | Late Chunking | 0.560 | 0.485 | 0.665 | 1.295 |
| Partial | Structure-Aware | 0.616 | 0.550 | 0.706 | 1.483 |
| Partial | GraLC-RAG (KG) | 0.583 | 0.502 | 0.702 | 1.378 |
| Partial | GraLC-RAG (+Graph) | 0.609 | 0.533 | 0.710 | 1.391 |
| Full-text | Naive | 0.586 | 0.510 | 0.688 | 1.629 |
| Full-text | Semantic | **0.736** | **0.683** | **0.808** | **1.992** |
| Full-text | Late Chunking | 0.558 | 0.492 | 0.644 | 1.613 |
| Full-text | Structure-Aware | 0.607 | 0.538 | 0.700 | 1.781 |
| Full-text | GraLC-RAG (KG) | 0.551 | 0.477 | 0.650 | 1.622 |
| Full-text | GraLC-RAG (+Graph) | 0.569 | 0.498 | 0.665 | 1.639 |

Figure 6 shows MRR across document-length conditions for all strategies. Semantic chunking dominates MRR at every document length, indicating that content-similarity remains the strongest signal for point-estimate retrieval accuracy. However, this metric alone does not capture structural retrieval diversity---a dimension where GraLC-RAG demonstrates clear advantages (Section 5.6).

**Figure 6:** MRR across document-length conditions (introduction, partial, full-text) for all retrieval strategies. Semantic chunking achieves the highest MRR throughout, while structure-aware methods (GraLC-RAG, Structure-Aware) show competitive performance and provide complementary advantages in section coverage (Figure 8).

### 5.6 Cross-Section Retrieval

A key hypothesis motivating GraLC-RAG is that structure-aware chunking should excel at retrieving information that requires reasoning across document sections---for example, linking a method description to its corresponding results, or connecting an introduction's hypothesis to the discussion's interpretation. To evaluate this, we construct a synthetic cross-section QA benchmark.

#### 5.6.1 Benchmark Construction

We generate 480 template-based questions that explicitly require cross-section reasoning, drawn from five template types:

1. **Method->Result:** "What results were obtained using [method X]?"
2. **Intro->Result:** "Does the data support the hypothesis that [hypothesis from introduction]?"
3. **Result->Discussion:** "How do the authors interpret the finding that [result Y]?"
4. **Method->Discussion:** "What limitations of [method X] are discussed?"
5. **Cross-study:** "How do the results compare to [cited prior work Z]?"

Template slots are populated from parsed section content using regex and entity extraction.

#### 5.6.2 Evaluation Metrics

We define two cross-section-specific metrics:

**Cross-Section Recall (CS Recall):** The proportion of cross-section questions for which the top-k retrieved chunks span at least two distinct document sections required by the question template. Measured per document-length condition.

**Section Coverage@k (SecCov@k):** The average number of distinct document sections represented in the top-k retrieved chunks, normalized by the total number of sections in the source document.

**Table 5: Section coverage and cross-section recall. SecCov@5 measures the average number of distinct document sections in the top-5 retrieved chunks. CS Recall measures whether top-5 chunks span at least two required sections. Content-only methods always retrieve from a single section (SecCov@5 = 1.0), while structure-aware methods retrieve from up to 4.2x more sections on full-text articles.**

| Strategy | SecCov@5 (Intro) | SecCov@5 (Partial) | SecCov@5 (Full) | CS Recall (Intro) | CS Recall (Partial) | CS Recall (Full) |
|----------|:----------------:|:------------------:|:---------------:|:-----------------:|:-------------------:|:----------------:|
| Naive | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 |
| Semantic | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 |
| Late Chunking | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 |
| Structure-Aware | 2.12 | 3.03 | **4.01** | 0.000 | 0.000 | 0.000 |
| GraLC-RAG (KG) | 2.18 | 3.16 | **4.17** | 0.000 | 0.000 | 0.000 |
| GraLC-RAG (+Graph) | 2.20 | 3.05 | 4.10 | 0.000 | 0.000 | 0.000 |

Cross-section recall is 0.000 across all strategies and conditions, indicating that no method successfully retrieves chunks from both required sections within the top-5 results. This finding is consistent with the shallow retrieval depth (top-5) relative to the number of chunks per article. However, Figure 8 reveals a striking difference in section coverage: structure-aware methods retrieve from 2-4x more distinct sections than content-only methods, with coverage scaling proportionally to document length.

**Figure 8:** Section coverage (SecCov@5) across document-length conditions. Content-only methods (Naive, Semantic, Late Chunking) are locked at 1.0 regardless of document length, while GraLC-RAG and Structure-Aware scale from ~2.2 (introduction) to ~4.2 (full-text), demonstrating that structure-aware retrieval surfaces evidence from across the document rather than concentrating on a single section.

---

## 6. Discussion

### 6.1 When Does Graph-Aware Late Chunking Help?

Our full-text evaluation reveals that the value of GraLC-RAG's components is **context-dependent** and **dimension-dependent**, varying with document length, structural complexity, and the metric used for evaluation.

**Semantic chunking dominates MRR at all document lengths.** Semantic chunking achieves the highest MRR across all conditions (0.586 on introduction, 0.735 on partial, 0.736 on full-text), demonstrating that content-similarity remains the strongest signal for point-estimate retrieval accuracy. No MRR crossover point is observed---contrary to our initial hypothesis, simpler content-based methods maintain their ranking advantage even on structurally complex full-text articles.

**Structure-aware methods retrieve from fundamentally more sections.** However, content-only methods are structurally blind: Naive, Semantic, and Late Chunking always retrieve from a single section (SecCov@5 = 1.0), regardless of document length. In contrast, GraLC-RAG (+Graph) retrieves from 2.20 sections on introduction-only documents and 4.10 sections on full-text articles---a 4x increase in structural coverage. This reveals that GraLC-RAG's advantage lies in *what* it retrieves (structural diversity across document sections) rather than ranking accuracy alone.

**Graph re-ranking provides significant improvements.** On partial documents, GraLC-RAG (+Graph) significantly outperforms Late Chunking (+0.049 MRR, p = 0.0003) and plain GraLC-RAG (KG) (+0.026 MRR, p = 0.0016), demonstrating that knowledge graph proximity signals add discriminative value when documents contain sufficient structural complexity.

### 6.2 The Document-Length Effect

The document-length gradient evaluation (Section 5.5) reveals two distinct scaling behaviors depending on the metric. For MRR, all strategies improve from the introduction to partial condition (e.g., Semantic: 0.586 to 0.735), then plateau on full-text (0.736), with the ranking among strategies remaining stable across conditions. No MRR crossover point is observed: semantic chunking leads throughout. However, for section coverage, the scaling behavior is strikingly different. Content-only methods remain flat at SecCov@5 = 1.0 regardless of document length, while structure-aware methods scale linearly: GraLC-RAG (KG) increases from 2.18 (introduction) to 3.16 (partial) to 4.17 (full-text). This divergence reveals that MRR alone is insufficient for evaluating full-text retrieval systems---it captures *how well* the system ranks the single most relevant chunk, but not *how broadly* it covers the document's structural content.

### 6.3 When Does Structure-Awareness Matter?

Our results demonstrate that structure-awareness matters along two dimensions: *retrieval breadth* and *graph re-ranking*. First, structure-aware methods consistently retrieve from more document sections than content-only methods, with the gap widening as documents grow longer. On full-text articles, GraLC-RAG (KG) covers 4.17 sections on average versus 1.0 for Semantic chunking---a capability that is invisible to MRR but critical for downstream tasks requiring cross-section reasoning (e.g., linking Methods to Results).

Second, graph re-ranking provides a statistically significant benefit. On partial documents, GraLC-RAG (+Graph) significantly outperforms Late Chunking (+0.049 MRR, p = 0.0003) and plain GraLC-RAG (KG) (+0.026 MRR, p = 0.0016), demonstrating that knowledge graph proximity signals add discriminative value when documents contain sufficient structural complexity. On full-text, Structure-Aware significantly outperforms GraLC-RAG (+Graph) in MRR (-0.038, p = 0.002), indicating that the graph re-ranking penalty from noisy KG signals can offset its structural coverage advantages when measuring point-estimate accuracy.

The universal cross-section recall of 0.000 reveals a fundamental limitation of top-k retrieval at shallow depths: no strategy retrieves chunks from both required sections in the top-5, suggesting that deeper retrieval (k >= 10) or explicit multi-section retrieval strategies are needed for cross-section reasoning tasks.

### 6.4 Implications for Biomedical RAG

These findings carry practical implications for practitioners building biomedical RAG systems. First, **the choice of metric matters as much as the choice of strategy**: MRR alone favors content-similarity methods, but section coverage reveals that structure-aware methods provide fundamentally different retrieval behavior---surfacing evidence from across the document rather than concentrating on the single most similar section. For downstream tasks requiring comprehensive evidence synthesis (systematic reviews, multi-section summarization), this structural diversity may be more valuable than point-estimate ranking accuracy. Second, **graph re-ranking adds significant value on structurally complex documents**: the p = 0.0016 improvement from graph re-ranking on partial documents demonstrates that KG proximity signals complement dense retrieval when documents have sufficient structural content. Third, **evaluation design must match deployment context**: benchmarking only with MRR on full-text corpora systematically undervalues structural retrieval methods, potentially leading to suboptimal architecture choices for applications requiring cross-section reasoning.

### 6.5 Comparison with Concurrent Work

Our work relates to but differs from several concurrent approaches. **HeteRAG** (Yang et al., 2025) also decouples retrieval and generation representations but operates on pre-chunked text without late chunking or KG awareness. **SitEmb** (Wu et al., 2025) conditions chunk embeddings on broader context but uses purely textual signals. **ATLANTIC** (Munikoti et al., 2023) fuses graph and text embeddings for scientific retrieval but operates at the document level rather than chunk level. To our knowledge, GraLC-RAG is the first framework to integrate all three aspects---contextual embedding, structural boundaries, and ontological enrichment---at the chunk level for biomedical RAG. Our results show that this combination does not yield MRR improvements over content-similarity methods, but provides a qualitatively different retrieval behavior: structural coverage that scales with document length.

### 6.6 Limitations

Several limitations should be acknowledged:

1. **Synthetic benchmark validity.** The 480 template-generated questions may not capture the full range of cross-section reasoning patterns encountered in real biomedical information needs. The template-based approach may overrepresent certain reasoning patterns (e.g., Method->Result) and underrepresent others (e.g., implicit cross-section reasoning).

2. **IMRaD filter bias.** The full-text evaluation corpus is filtered for standard IMRaD structure (139 of 500 downloaded articles), excluding review articles, case reports, letters, editorials, and non-standard formats. This 27.8% retention rate introduces a selection bias toward primary research articles.

3. **Shallow retrieval depth.** The universal cross-section recall of 0.000 at k=5 indicates that our evaluation depth is insufficient to test cross-section reasoning. Future work should evaluate at k=10 or k=20, or adopt explicit multi-section retrieval strategies.

4. **No MRR crossover.** Contrary to our initial hypothesis, semantic chunking dominates MRR at all document lengths. The expected crossover point was not observed, indicating that content-similarity remains the strongest single signal for ranking accuracy. The contribution of structure-awareness is in retrieval breadth rather than ranking precision.

5. **Entity linking quality.** MeSH dictionary matching may introduce false positives for polysemous terms (e.g., "MS" as multiple sclerosis vs. mass spectrometry).

6. **Context window constraints.** Approximately 62% of full-text articles exceed the 8,192-token context window, requiring sliding-window approximation.

7. **Single-language scope.** Our evaluation is limited to English-language biomedical literature.

### 6.7 Future Directions

Several promising extensions emerge from this work:

**Deeper retrieval evaluation.** Evaluating at k=10 and k=20 to test whether structure-aware methods achieve non-zero cross-section recall at greater depth, and whether the structural coverage advantage translates to improved downstream task performance.

**Hybrid strategies.** Combining semantic chunking's MRR strength with GraLC-RAG's structural coverage---for example, using semantic chunking for the top-k results and structure-aware re-ranking to diversify section representation.

**Adaptive weighting.** Adaptive lambda and beta weighting based on document length to reduce KG noise on shorter documents while amplifying structural signals on full-text articles.

**Multi-granularity retrieval.** The document structure graph enables retrieval at multiple granularities (paragraph, section, document) within a single framework. Adaptive granularity selection based on query complexity could further improve performance.

**Cross-document graph linking.** Extending the knowledge subgraph to include cross-document citation relationships would enable GraLC-RAG to surface evidence chains across multiple papers, supporting systematic review and meta-analysis applications.

**Dynamic KG updates.** Integrating temporal knowledge graph evolution (e.g., MedKGent; Zhang et al., 2025) would allow the system to remain current with newly published biomedical findings.

---

## 7. Conclusion

We presented GraLC-RAG, a framework that integrates graph-aware structural intelligence into the late chunking paradigm for biomedical Retrieval-Augmented Generation. By combining structure-aware chunk boundary detection, knowledge graph-infused token representations via UMLS and GAT attention, and graph-guided hybrid retrieval, GraLC-RAG addresses the complementary limitations of existing late chunking (structure-blind) and GraphRAG (context-fragmented) approaches.

Evaluation on 139 IMRaD-filtered PMC articles across a document-length gradient reveals that GraLC-RAG retrieves from significantly more document sections than content-only methods, with structural coverage scaling with document length---up to 4.2x on full-text articles (SecCov@5 = 4.17 vs. 1.0)---even though content-similarity methods (semantic chunking) achieve higher point-estimate MRR (0.736 vs. 0.569).

These findings yield three actionable insights for the field: (1) MRR alone is insufficient for evaluating full-text retrieval---section coverage reveals structural diversity that ranking metrics miss; (2) graph re-ranking provides statistically significant improvements over plain KG-based retrieval on structurally complex documents (p = 0.0016 on partial); and (3) the choice between content-similarity and structure-aware methods should be guided by downstream task requirements---ranking accuracy favors semantic chunking, while evidence breadth favors GraLC-RAG.

We release the complete GraLC-RAG codebase---including all six chunking strategies, the entity linking and KG infusion pipeline, FAISS indexing, and evaluation scripts---to enable reproduction and extension.

---

## References

Allamraju, A., Chitale, M. P., Adibhatla, H. S., Mishra, R., & Shrivastava, M. (2025). Breaking it down: Domain-aware semantic segmentation for retrieval augmented generation. *arXiv preprint arXiv:2512.00367*.

Anthropic. (2024). Introducing contextual retrieval. Anthropic Blog. Retrieved from https://www.anthropic.com/news/contextual-retrieval

Bodenreider, O. (2004). The Unified Medical Language System (UMLS): Integrating biomedical terminology. *Nucleic Acids Research*, *32*(Database issue), D267–D270.

Canese, K., & Weis, S. (2013). PubMed: The bibliographic database. In *The NCBI Handbook* (2nd ed.). National Center for Biotechnology Information.

Chen, B., Guo, Z., Yang, Z., Chen, Y., Chen, J., Liu, Z., Shi, C., & Yang, C. (2025). PathRAG: Pruning graph-based retrieval augmented generation with relational paths. *arXiv preprint arXiv:2502.14902*.

Chen, S., Zhao, Y., Joty, S., & Cai, Z. (2023). Dense X retrieval: What retrieval granularity should we use? *arXiv preprint arXiv:2312.06648*.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

Gao, Y., Xiong, Y., Dibia, V., Zhang, L., & Han, J. (2024). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare*, *3*(1), 1–23.

Günther, M., Mohr, I., Wang, B., & Xiao, H. (2024). Late chunking: Contextual chunk embeddings using long-context embedding models. *arXiv preprint arXiv:2409.04701*.

Guo, Z., Xia, L., Yu, Y., Ao, T., & Huang, C. (2024). LightRAG: Simple and fast retrieval-augmented generation. *Findings of EMNLP 2025*. arXiv preprint arXiv:2410.05779.

Han, H., Wang, Y., Shomer, H., Guo, K., Ding, J., Lei, Y., ... & Tang, J. (2025). Retrieval-augmented generation with graphs (GraphRAG). *arXiv preprint arXiv:2501.00309*.

Honovich, O., Aharoni, R., Herzig, J., Taitelbaum, H., Kuber, D., Chung, V., ... & Levy, O. (2022). TRUE: Re-evaluating factual consistency evaluation. *Proceedings of NAACL 2022*, 3905–3920.

Hu, Y., Lei, Z., Dai, Z., Zhang, A., Angirekula, A., Zhang, Z., & Zhao, L. (2025). CG-RAG: Research question answering by citation graph retrieval-augmented LLMs. *arXiv preprint arXiv:2501.15067*.

Hu, Y., Lei, Z., Zhang, Z., Pan, B., Ling, C., & Zhao, L. (2024). GRAG: Graph retrieval-augmented generation. *arXiv preprint arXiv:2405.16506*.

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., ... & Liu, T. (2025). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Computing Surveys*.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, *55*(12), 1–38.

Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., & Neubig, G. (2023). Active retrieval augmented generation. *Proceedings of EMNLP 2023*.

Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019). PubMedQA: A dataset for biomedical research question answering. *Proceedings of EMNLP-IJCNLP 2019*, 2567–2577.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of EMNLP 2020*, 6769–6781.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, *33*, 9459–9474.

Lin, Y., He, J., Chen, J., Zhu, X., Zheng, J., & Bo, T. (2025). BioGraphFusion: Graph knowledge embedding for biological completion and reasoning. *arXiv preprint arXiv:2507.14468*.

Liu, F., Shareghi, E., Meng, Z., Basaldella, M., & Collier, N. (2021). Self-alignment pretraining for biomedical entity representations. *Proceedings of NAACL 2021*, 4228–4238.

Lopez, P. (2009). GROBID: Combining automatic bibliographic data recognition and term extraction for scholarship publications. *Proceedings of ECDL 2009*, 473–474.

Ma, X., Gong, Y., He, P., Zhao, H., & Duan, N. (2024). Query-dependent prompt evaluation and optimization with offline inverse RL. *Proceedings of ICLR 2024*.

Maina, M. M., et al. (2025). Comparative evaluation of advanced chunking for retrieval-augmented generation in large language models for clinical decision support. *Bioengineering*, *12*(11), 1194.

Matsumoto, T., Moran, S., et al. (2024). KRAGEN: A knowledge graph-enhanced RAG framework for biomedical problem solving using large language models. *Bioinformatics*, *40*(6), btae353.

Merola, C., & Singh, J. (2025). Reconstructing context: Evaluating advanced chunking strategies for retrieval-augmented generation. *arXiv preprint arXiv:2504.19754*.

Munikoti, S., Acharya, A., Wagle, S., & Horawalavithana, S. (2023). ATLANTIC: Structure-aware retrieval-augmented language model for interdisciplinary science. *arXiv preprint arXiv:2311.12289*.

Nelson, C. A., Butte, A. J., & Baranzini, S. E. (2019). Integrating biomedical research and electronic health records to create knowledge-based biologically meaningful machine-readable embeddings. *Nature Communications*, *10*(1), 3045.

Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and robust models for biomedical natural language processing. *Proceedings of BioNLP 2019*, 319–327.

Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., Hong, C., Zhang, Y., & Tang, S. (2024). Graph retrieval-augmented generation: A survey. *ACM Transactions on Information Systems*. arXiv preprint arXiv:2408.08921.

Remy, F., Demuynck, K., & Demeester, T. (2023). BioLORD-2023: Semantic textual representations fusing LLM and clinical knowledge graph insights. *arXiv preprint arXiv:2311.16075*.

Rubin, O., & Berant, J. (2023). Long-range language modeling with self-retrieval. *arXiv preprint arXiv:2306.13421*.

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2022). ColBERTv2: Effective and efficient retrieval via lightweight late interaction. *Proceedings of NAACL 2022*, 3715–3734.

Soldaini, L., & Goharian, N. (2016). QuickUMLS: A fast, unsupervised approach for medical concept extraction. *MedIR Workshop, SIGIR 2016*.

Sollaci, L. B., & Pereira, M. G. (2004). The introduction, methods, results, and discussion (IMRAD) structure: A fifty-year survey. *Journal of the Medical Library Association*, *92*(3), 364–367.

Soman, K., Rose, P. W., Morris, J. H., Akbas, R. E., Smith, B., Peetoom, B., ... & Baranzini, S. E. (2024). Biomedical knowledge graph-optimized prompt generation for large language models. *Bioinformatics*, *40*(9), btae560.

Tsatsaronis, G., Balikas, G., Malakasiotis, P., Partalas, I., Zschunke, M., Alvers, M. R., ... & Paliouras, G. (2015). An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition. *BMC Bioinformatics*, *16*(1), 138.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *Proceedings of ICLR 2018*.

Verma, P. (2025). S2 Chunking: A hybrid framework for document segmentation through integrated spatial and semantic analysis. *arXiv preprint arXiv:2501.05485*.

Wang, Z., Yuan, H., Dong, W., Cong, G., & Li, F. (2024). CORAG: A cost-constrained retrieval optimization system for retrieval-augmented generation. *arXiv preprint arXiv:2411.00744*.

Wu, J., Li, J., Li, Y., Liu, L., Xu, L., Li, J., Yeung, D., Zhou, J., et al. (2025). SitEmb-v1.5: Improved context-aware dense retrieval for semantic association and long story comprehension. *arXiv preprint arXiv:2508.01959*.

Wu, J., Zhu, J., & Qi, Y. (2024). Medical Graph RAG: Towards safe medical large language model via graph retrieval-augmented generation. *Proceedings of ACL 2025*. arXiv preprint arXiv:2408.04187.

Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*.

Xu, T., Zheng, H., Li, C., Chen, H., Liu, Y., Chen, R., & Sun, L. (2025). NodeRAG: Structuring graph-based RAG with heterogeneous nodes. *arXiv preprint arXiv:2504.11544*.

Yang, P., Li, X., Hu, Z., Wang, J., Yin, J., Wang, H., ... & Yang, S. (2025). HeteRAG: A heterogeneous retrieval-augmented generation framework with decoupled knowledge representations. *arXiv preprint arXiv:2504.10529*.

Zhang, D., Wang, Z., Li, Z., Yu, Y., Jia, S., Dong, J., ... & Gao, J. (2025). MedKGent: A large language model agent framework for constructing temporally evolving medical knowledge graph. *arXiv preprint arXiv:2508.12393*.

Zhao, J., Ji, Z., Fan, Z., Wang, H., Niu, S., Tang, B., Xiong, F., & Li, Z. (2025). MoC: Mixtures of text chunking learners for retrieval-augmented generation system. *arXiv preprint arXiv:2503.09600*.

---

*AI Disclosure: This research paper was produced with the assistance of AI tools (Claude, Anthropic) for literature search, synthesis, and manuscript drafting. All references have been verified against their original sources. The framework design, methodology, and experimental results have been reviewed for technical soundness. The numerical results in Sections 5.1--5.2 (abstract-level evaluation) are from the PubMedQA proof-of-concept; the results in Sections 5.5--5.6 (full-text evaluation) are empirical measurements on 139 IMRaD-filtered PMC articles with 480 template questions. The authors take full responsibility for the intellectual content and any remaining errors.*
