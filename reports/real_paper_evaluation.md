# Real Paper Evaluation

Generated at: `2026-06-18T08:38:54.212949+00:00`

## Configuration

- Embedding backend: `sentence-transformers/all-MiniLM-L6-v2`
- LLM provider: `mock`
- Vector store: `InMemoryVectorStore` using the same retrieval interface as the app
- Corpus source: real arXiv PDFs parsed locally with PyMuPDF

## Corpus

| paper id | title | pages | chunks |
| --- | --- | ---: | ---: |
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 19 | 20 |
| 2404.16130 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | 26 | 31 |
| 2309.15217 | Ragas: Automated Evaluation of Retrieval Augmented Generation | 8 | 10 |

## Metrics

- Average Recall@3: 1.00
- Average Recall@5: 1.00
- Citation coverage: 6/6

## Query Results

| question | expected papers | retrieved @3 | Recall@3 | Recall@5 | citations | confidence |
| --- | --- | --- | ---: | ---: | --- | --- |
| What is the main idea behind retrieval augmented generation? | 2005.11401 | 2309.15217, 2404.16130, 2005.11401 | 1.00 | 1.00 | yes | Medium |
| How does GraphRAG use graph structure for global sensemaking? | 2404.16130 | 2404.16130, 2404.16130, 2404.16130 | 1.00 | 1.00 | yes | Medium |
| What does RAGAS evaluate in retrieval augmented generation systems? | 2309.15217 | 2309.15217, 2309.15217, 2404.16130 | 1.00 | 1.00 | yes | Medium |
| Which paper discusses answer faithfulness and citation-style evaluation? | 2309.15217 | 2309.15217, 2404.16130, 2309.15217 | 1.00 | 1.00 | yes | Medium |
| Which paper introduced retrieval-augmented generation for knowledge-intensive NLP? | 2005.11401 | 2005.11401, 2309.15217, 2309.15217 | 1.00 | 1.00 | yes | High |
| Which paper explains graph-based retrieval augmented generation? | 2404.16130 | 2404.16130, 2404.16130, 2309.15217 | 1.00 | 1.00 | yes | Medium |

## Interpretation

This run evaluates retrieval and citation formatting on real PDF text. The generation provider is the deterministic `mock` client so the reported answer-quality metric is citation coverage rather than semantic answer quality. Use Ollama or an OpenAI-compatible provider for qualitative answer evaluation.
