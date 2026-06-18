# System Limitations

- PDF extraction can fail on scanned papers without OCR.
- Word-based chunking is transparent but less precise than tokenizer-aware chunking.
- Retrieval quality depends on the embedding model and the indexed paper set.
- The app can only answer from uploaded and indexed content.
- The mock LLM is useful for tests but does not perform real reasoning.
- Chroma indexes are local artifacts and are intentionally ignored by Git.

## Future Work

- Add OCR for scanned PDFs.
- Add Semantic Scholar metadata.
- Add BibTeX or Zotero export.
- Add citation graph exploration.
- Add Docker packaging.
- Compare multiple embedding models on the same question set.
