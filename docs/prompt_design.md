# Prompt Design

## RAG Answer Prompt

The RAG prompt instructs the model to answer only from retrieved context, cite evidence, admit insufficient evidence, and avoid invented sources. It requires this response structure:

```text
Answer:
...

Evidence:
[1] ...

Confidence:
High / Medium / Low
```

The key safety rules are:

- Only answer based on the provided context.
- Always cite evidence using the provided source numbers.
- If the context is insufficient, say so clearly.
- Do not invent sources, citations, paper claims, datasets, or results.

## Summarisation Prompt

The summarisation prompt asks for structured fields:

- Research Problem
- Method
- Dataset / Experiment
- Key Findings
- Limitations
- Potential Applications

If the text does not clearly support a field, the model must output:

```text
Not clearly stated in the provided text
```

## Paper Comparison Prompt

The comparison prompt asks for a Markdown table with:

```text
paper title | research problem | method | dataset | strength | limitation | possible extension
```

It repeats the rule that unsupported fields must not be invented.

## Insufficient Evidence Prompt

When no chunks are retrieved, the pipeline does not call the LLM. It returns a deterministic low-confidence response explaining that the current paper library is insufficient.
