# Research: Pre-Generated QA Pairs & Fixed Question Categories for RAG

Two questions Andrew wanted evidence on:

1. **Is pre-generating QA pairs (instead of chunking raw docs) a good idea for RAG?**
2. **Is constraining the LLM with fixed question categories (instead of letting it freestyle) a good idea?**

Short answer to both: yes, well-supported. Details and links below.

---

## Part 1: Pre-Generated QA Pairs as a RAG Strategy

Our system pre-generates Q&A pairs from structured MCP data at index time, rather than chunking raw documents and doing similarity search at query time. This is a validated pattern.

### Key research and industry practice

**doc2query / docTTTTTquery (Microsoft Research, 2019)**
The foundational work. For each document, generate synthetic questions it could answer, then append them to the document before indexing. On MS MARCO benchmarks, dramatically improved retrieval while keeping query-time cost low — expensive neural inference is pushed to index time.
- Paper: [From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)
- Code: [castorini/docTTTTTquery](https://github.com/castorini/docTTTTTquery)

**LlamaIndex QuestionsAnsweredExtractor (production framework)**
LlamaIndex — one of the two dominant RAG frameworks — has this as a built-in, first-class feature. Their `QuestionsAnsweredExtractor` generates questions each chunk can answer, stores them as metadata, and uses question-to-question matching at retrieval time.
- Docs: [QuestionsAnsweredExtractor API](https://developers.llamaindex.ai/python/framework-api-reference/extractors/question/)
- Guide: [Building and Evaluating a QA System with LlamaIndex](https://www.llamaindex.ai/blog/building-and-evaluating-a-qa-system-with-llamaindex-3f02e9d87ce1)

**Anthropic's Contextual Retrieval (Sept 2024)**
Validates the principle that LLM-powered pre-processing at index time dramatically improves RAG. Reported a **49% reduction in retrieval failures** by using Claude to add context to chunks before embedding.
- Blog: [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

**RAFT: Retrieval Augmented Fine Tuning (UC Berkeley, 2024)**
Generates synthetic QA pairs from domain documents to fine-tune models for domain-specific RAG. Validates that LLM-generated QA from source documents is a sound strategy.
- Paper: [arxiv.org/abs/2403.10131](https://arxiv.org/abs/2403.10131)
- Blog: [Gorilla RAFT](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)

**SimRAG (NAACL 2025)**
Self-improving RAG through synthetic QA pairs — generates pseudo-labeled QA tuples by extracting candidate answers then generating questions conditioned on document + answer.
- Paper: [SimRAG at NAACL 2025](https://aclanthology.org/2025.naacl-long.575.pdf)

### Why pre-generated QA is especially strong for our use case

| Advantage | Why it applies to us |
|---|---|
| **Better semantic matching** | User questions match against pre-generated questions (question-to-question similarity), which is an easier retrieval problem than question-to-passage. |
| **Controlled answer quality** | Answers generated at index time with full entity context, reviewable before serving. |
| **Citation precision** | Each QA pair maps to one entity with a `<<SRC:domain:entity_id>>` citation. Traditional RAG struggles with attribution. |
| **Structured data advantage** | Our source data is structured JSON from MCP APIs. Chunking structured data is a well-known antipattern — loses field names, relationships, context boundaries. ([NVIDIA blog](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/), [Stack Overflow blog](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)) |
| **Deterministic coverage** | We know exactly what questions our system can answer. Chunking-based RAG depends on the retriever finding the right chunk, which is probabilistic. |
| **Lower query-time cost** | No LLM call needed at query time for answer generation — just retrieval. The expensive LLM work is amortized at index time. |

### Tradeoffs

| Tradeoff | Our mitigation |
|---|---|
| Upfront compute cost | ~$19 for all ~9,150 entities with gpt-4o-mini. Pay once. |
| Can only answer anticipated questions | `ComparisonGenerator` handles cross-entity queries; diverse question categories cover the known information needs. |
| Staleness when source data changes | HPC resource specs don't change frequently; re-extraction is cheap. |
| Potential hallucination at index time | Citation validator checks `<<SRC:...>>` markers against real MCP entities. Source data attached to each QAPair for human review in Argilla. |

---

## Part 2: Fixed Question Categories vs. LLM Freestyle

This is the more specific question: should we tell the LLM "generate one QA per category from this list: overview, hardware, people, access..." or just say "generate whatever QA pairs you think are good"?

### Key research

**KGQuest: Template-Driven QA from Knowledge Graphs (Nov 2025)**
Almost exactly our pattern — structured data (knowledge graph), template-driven question generation by relation/entity type, one template per category. Template-driven was **1,000x faster** (9 minutes vs 160 hours for pure LLM generation on 367K questions) with 80-90% correctness. A lightweight LLM pass then refines fluency — not deciding *what* to ask.
- Paper: [KGQuest](https://arxiv.org/html/2511.11258)

**DataMorgana: Combinatorial QA with Fixed Categories (Jan 2025, SIGIR 2025)**
Defines fixed **question types** (factoid, open-ended, premise-based, etc.) and **user types** (expert, novice, domain-specific roles) then generates QA for every combination. Defining categories up front produces dramatically better **lexical, syntactic, and semantic diversity** than freeform generation. Used in the SIGIR 2025 LiveRAG competition as the standard approach.
- Paper: [DataMorgana](https://arxiv.org/abs/2501.12789)

**RAGen: Bloom's Taxonomy as Question Categories (Oct 2025)**
Uses Bloom's cognitive levels as fixed categories to guide generation. Result: **"markedly richer mix of higher-order question types"** vs letting the LLM freestyle, which overwhelmingly produces low-level recall questions. Models fine-tuned on categorized output outperformed freeform baselines on ROUGE-L and BERT-F1.
- Paper: [RAGen](https://arxiv.org/html/2510.11217)

**Bloom's Taxonomy Question Generation (ACL/NeurIPS 2023-2024)**
Multiple papers show LLMs told "generate questions" without categories heavily skew toward simple factual recall. With explicit categories, diversity and quality improve significantly.
- Paper: [Automated Educational QG at Different Bloom's Skill Levels](https://arxiv.org/abs/2408.04394)
- Workshop: [NeurIPS 2023: Aligning with Bloom's Taxonomy](https://neurips.cc/virtual/2023/79098)

**Structured prompting research (general)**
Structured prompts with explicit categories reduce hallucination, improve consistency, and eliminate the problem where accuracy of question types decreases as the sequence progresses (i.e., freeform LLMs start strong then get repetitive/lazy).
- Overview: [Structured Prompting Techniques](https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/)

### Categories vs. freestyle comparison

| | Freeform ("generate QA pairs") | Fixed categories ("one QA per: overview, hardware, people, access") |
|---|---|---|
| **Coverage** | LLM decides what to cover, skews toward surface-level | Every category guaranteed to be attempted |
| **Diversity** | Repetitive within a batch; quality drops over sequence | Forced diversity by construction |
| **Consistency across entities** | Delta gets 9 questions, Bridges-2 gets 4 | Every entity gets the same categories |
| **IDs** | Slug from question text (fragile, changes across runs) | `{domain}_{entity}_{category}` (stable) |
| **Reviewability** | Hard to compare across entities | Apples-to-apples comparison |
| **Cost control** | Unpredictable token usage | Predictable: N categories x M entities |

---

## Summary

> Pre-generating QA pairs from structured data instead of chunking is a validated strategy (doc2query, LlamaIndex, Anthropic Contextual Retrieval). Constraining the LLM with fixed question categories instead of letting it freestyle is also well-supported (KGQuest, DataMorgana/SIGIR 2025, RAGen, Bloom's taxonomy papers). Our approach combines both patterns, and is especially well-suited for structured MCP data where traditional chunking is known to perform poorly.

### Best links to share

| For what | Link |
|---|---|
| Pre-generated QA is an industry standard | [LlamaIndex QuestionsAnsweredExtractor](https://developers.llamaindex.ai/python/framework-api-reference/extractors/question/) |
| LLM-at-index-time works | [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) |
| Template-driven QA from structured data | [KGQuest](https://arxiv.org/html/2511.11258) |
| Fixed categories = better diversity | [DataMorgana (SIGIR 2025)](https://arxiv.org/abs/2501.12789) |
| Category-guided > freeform on benchmarks | [RAGen](https://arxiv.org/html/2510.11217) |
