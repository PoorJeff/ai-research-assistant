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

## Current MVP Status

The repository ships the evaluation helpers and a set of ten demo questions. After indexing a real paper set, run the demo questions and record observed Recall@3, Recall@5, citation coverage, and unsupported-claim notes in this file.
