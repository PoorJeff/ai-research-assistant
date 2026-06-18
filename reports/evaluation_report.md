# Evaluation Report

## Purpose

The MVP includes lightweight evaluation to show that the system is more than a chat interface. The goal is to inspect whether retrieval returns relevant evidence and whether generated answers remain grounded in that evidence.

## Retrieval Evaluation

For each test question, manually define one or more expected paper IDs, paper titles, or keywords. Run retrieval and compute:

```text
Recall@k = relevant retrieved items in top k / expected relevant items
```

The helper `compute_recall_at_k` supports Recall@3 and Recall@5 style checks.

## Answer Evaluation

Each answer is checked for:

- presence of an `Evidence:` section,
- numbered citations such as `[1]`,
- refusal or low-confidence behavior when no evidence is retrieved,
- obvious claims that go beyond retrieved context.

## Real Paper Run

A reproducible real-paper run is available in `reports/real_paper_evaluation.md` and `reports/real_paper_evaluation.json`.

Configuration:

- Corpus: 3 real arXiv PDFs.
- Papers:
  - `2005.11401`: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
  - `2404.16130`: From Local to Global: A Graph RAG Approach to Query-Focused Summarization.
  - `2309.15217`: Ragas: Automated Evaluation of Retrieval Augmented Generation.
- PDF parser: PyMuPDF.
- Chunking: 700 words with 120-word overlap.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- LLM provider: deterministic `mock` client, so this run evaluates retrieval and citation formatting rather than qualitative answer fluency.

Results from the latest run:

| metric | result |
| --- | ---: |
| Papers parsed | 3 |
| Total pages parsed | 53 |
| Total chunks indexed | 61 |
| Evaluation questions | 6 |
| Average Recall@3 | 1.00 |
| Average Recall@5 | 1.00 |
| Citation coverage | 6/6 |

This moves the project beyond a UI demo: the retrieval layer has been exercised on actual academic PDFs, and the answer pipeline consistently returns numbered evidence for the evaluated questions.

## Reproducing The Evaluation

Run:

```bash
python scripts/run_real_paper_evaluation.py --embedding-backend sentence-transformer
```

The script downloads PDFs into `data/papers/real_demo/`, which is ignored by Git, then writes:

- `reports/real_paper_evaluation.md`
- `reports/real_paper_evaluation.json`
