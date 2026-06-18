# AI Research Assistant - 项目要求与 Pipeline

## 1. 项目定位

这个项目用于申请 Master of Artificial Intelligence、Master of Data Science、Master of Information Technology、Master of Computer Science 等专业。目标是构建一个通用 AI 学术研究助手，帮助用户检索论文、理解论文、比较论文，并基于文献内容进行可追溯问答。

项目不绑定医疗方向，建议覆盖通用 AI / Data Science / Computer Science 论文。

推荐项目标题：

```text
AI Research Assistant: Literature Search, Paper Summarisation and Retrieval-Augmented Q&A
```

## 2. 申请加分点

这个项目应证明你具备以下能力：

- 能构建 LLM application，而不是只调用聊天接口。
- 能处理 PDF、论文元数据和非结构化文本。
- 能实现检索增强生成，即 RAG。
- 能设计 chunking、embedding、vector search 和 citation-based answer generation。
- 能用评估指标检查检索质量和回答质量。
- 能把 AI 系统做成可运行 demo。
- 能写清楚系统限制、风险和改进方向。

这个项目对 AI 专业尤其有帮助，因为它展示了 LLM、信息检索、NLP、数据工程和软件工程的结合。

## 3. 推荐技术栈

核心技术：

```text
Python
Streamlit
LangChain 或 LlamaIndex
ChromaDB
sentence-transformers
PyMuPDF 或 pypdf
arxiv API
pandas
pytest
```

LLM 选择：

```text
Ollama local model：推荐用于本地演示
Hugging Face model：可选
OpenAI-compatible API：可选，但不作为必须依赖
```

Embedding 选择：

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-small-en-v1.5
intfloat/e5-small-v2
```

MVP 优先使用轻量 embedding 模型，保证普通电脑可以运行。

## 4. 数据来源

首选数据来源：

```text
arXiv API
用户上传 PDF
```

推荐默认主题：

```text
retrieval augmented generation
knowledge graph question answering
large language model evaluation
data science automation
```

系统应允许用户输入自定义关键词，例如：

```text
"GraphRAG"
"LLM evaluation"
"time series forecasting"
"computer vision self-supervised learning"
```

## 5. 仓库结构要求

建议仓库结构：

```text
ai-research-assistant/
├── README.md
├── pyproject.toml 或 requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── arxiv_search.py
│   ├── pdf_loader.py
│   ├── text_cleaning.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── summarizer.py
│   ├── rag_pipeline.py
│   ├── paper_compare.py
│   └── evaluation.py
├── data/
│   ├── papers/
│   └── README.md
├── vectorstore/
│   └── .gitkeep
├── reports/
│   ├── demo_queries.md
│   ├── evaluation_report.md
│   └── system_limitations.md
├── tests/
│   ├── test_chunking.py
│   ├── test_pdf_loader.py
│   ├── test_retrieval.py
│   └── test_rag_pipeline.py
└── docs/
    ├── system_design.md
    ├── prompt_design.md
    └── model_card.md
```

## 6. Pipeline 设计

### Step 1: User Input

系统支持两种输入方式：

```text
1. 用户输入研究主题，系统从 arXiv 搜索论文
2. 用户上传 PDF，系统解析并加入本地知识库
```

要求：

- 用户可以设置搜索关键词。
- 用户可以设置最多检索论文数量，如 5、10、20。
- 用户可以选择是否下载 PDF。

输出：

```text
paper metadata table
paper title
authors
published date
abstract
arXiv URL
PDF URL
```

### Step 2: Paper Search and Metadata Collection

要求：

- 使用 arXiv API 查询论文。
- 保存论文元数据到 DataFrame。
- 支持按 relevance 或 submitted date 排序。
- 保存元数据到 CSV。

输出：

```text
data/papers/metadata.csv
src/arxiv_search.py
```

### Step 3: PDF Loading and Text Extraction

要求：

- 支持从本地 PDF 提取文本。
- 支持解析标题、页码和正文。
- 去除过多换行、页眉页脚和明显乱码。
- 如果 PDF 解析失败，要返回清晰错误信息。

输出：

```text
src/pdf_loader.py
src/text_cleaning.py
```

### Step 4: Text Chunking

要求：

- 将论文文本切分为适合 embedding 的 chunks。
- 每个 chunk 保留元数据：

```text
paper_title
paper_id
page_number
section if available
chunk_id
source_url
```

推荐策略：

```text
chunk_size: 500-800 tokens
chunk_overlap: 80-120 tokens
```

必须在文档中解释为什么选择这个 chunking 策略。

输出：

```text
src/chunking.py
docs/system_design.md
```

### Step 5: Embedding and Vector Store

要求：

- 使用 sentence-transformers 生成 embeddings。
- 使用 ChromaDB 存储向量。
- 支持重建索引。
- 支持增量添加新论文。

输出：

```text
src/embeddings.py
src/vector_store.py
vectorstore/
```

### Step 6: Summarisation

系统应支持对单篇论文生成结构化摘要。

摘要字段：

```text
Research Problem
Method
Dataset / Experiment
Key Findings
Limitations
Potential Applications
```

要求：

- 摘要不能只是一段自由文本。
- 摘要必须尽量基于论文原文。
- 如果无法确定某项内容，应输出 "Not clearly stated in the provided text"。

输出：

```text
src/summarizer.py
```

### Step 7: Paper Comparison

系统应支持对多篇论文进行对比。

对比字段：

```text
paper title
research problem
method
dataset
strength
limitation
possible extension
```

输出形式：

```text
Markdown table
Streamlit table
CSV export
```

输出：

```text
src/paper_compare.py
```

### Step 8: RAG Question Answering

用户可以向本地论文库提问。

要求：

- 对问题进行 embedding。
- 从 ChromaDB 检索 top-k chunks。
- 将 chunks 和问题输入 LLM。
- 回答必须带引用。
- 如果检索内容不足，应回答无法从当前资料确认。

回答格式：

```text
Answer:
...

Evidence:
[1] Paper title, page/chunk, source URL
[2] Paper title, page/chunk, source URL

Confidence:
High / Medium / Low
```

禁止：

```text
没有 evidence 的确定性回答
编造论文结论
编造引用
```

输出：

```text
src/rag_pipeline.py
```

### Step 9: Evaluation

必须做一个轻量评估，而不是只展示 demo。

检索评估：

```text
Top-k retrieved chunks 是否包含相关论文
Recall@3
Recall@5
人工标注 10 个测试问题
```

回答评估：

```text
答案是否引用来源
答案是否超出证据范围
答案是否包含明显 hallucination
```

输出：

```text
reports/evaluation_report.md
src/evaluation.py
```

### Step 10: Streamlit Demo

界面至少包含四个页面或区域：

```text
1. Search Papers
2. Upload PDFs
3. Ask Questions
4. Compare Papers
```

必须显示：

```text
检索到的论文列表
论文摘要
RAG 回答
引用来源
检索到的 chunks
```

输出：

```text
app/streamlit_app.py
```

## 7. Prompt 设计要求

必须在 `docs/prompt_design.md` 中记录 prompts。

至少包含：

```text
summarisation prompt
paper comparison prompt
RAG answer prompt
refusal / insufficient evidence prompt
```

RAG answer prompt 必须强调：

```text
Only answer based on the provided context.
Always cite evidence.
If the context is insufficient, say so.
Do not invent sources.
```

## 8. README 要求

README 必须包含：

```text
Project Overview
Why this project matters
Features
System Architecture
Pipeline
Screenshots
Example queries
Evaluation results
How to run locally
Limitations
Future work
```

README 中建议放一张架构图：

```text
User Query
-> Paper Search / PDF Upload
-> Text Extraction
-> Chunking
-> Embedding
-> ChromaDB
-> Retrieval
-> LLM Answer
-> Evidence-based Response
```

## 9. 验收标准

项目完成标准：

- 可以从 arXiv 搜索论文并保存元数据。
- 可以解析本地 PDF。
- 可以建立 Chroma 向量库。
- 可以对论文进行结构化摘要。
- 可以对多篇论文进行对比。
- 可以基于论文库进行 RAG 问答。
- 回答必须包含 evidence。
- Streamlit demo 可以运行。
- 至少有 10 个测试问题和评估结果。
- README 有截图、运行方式和结果展示。
- tests 至少覆盖 chunking、PDF loading、retrieval。

## 10. 时间安排

建议 14-18 天完成：

```text
Day 1: 仓库结构、README 初版、系统设计
Day 2-3: arXiv search 和 metadata pipeline
Day 4-5: PDF loading 和 text cleaning
Day 6: chunking
Day 7-8: embedding 和 ChromaDB
Day 9: summarisation
Day 10: paper comparison
Day 11-12: RAG Q&A
Day 13: Streamlit demo
Day 14: evaluation
Day 15: tests
Day 16-18: README、截图、报告、代码清理
```

## 11. 可写进 CV 的项目描述

中文版本：

```text
构建通用 AI 学术研究助手，支持 arXiv 论文检索、本地 PDF 解析、结构化摘要、多论文对比和基于 ChromaDB 的 RAG 问答；设计 chunking、embedding、向量检索和 evidence-based answer generation 流程，并通过 Recall@k 和人工测试问题评估检索与回答质量。
```

英文版本：

```text
Built an AI research assistant for literature search, PDF parsing, structured paper summarisation, multi-paper comparison and retrieval-augmented question answering. Implemented chunking, embedding, ChromaDB-based retrieval and evidence-grounded answer generation, with evaluation using Recall@k and manually curated research questions.
```

## 12. 加分增强项

如果时间充足，可以增加：

- 支持 Semantic Scholar API。
- 支持 Zotero / BibTeX export。
- 支持 citation graph。
- 支持论文方法/数据集自动抽取。
- 支持多 embedding 模型对比。
- 支持本地 Ollama 模型和 API 模型切换。
- Docker 一键运行。
- GitHub Actions 自动测试。

优先级：

```text
高：RAG Q&A、citation、Streamlit demo、README 截图
中：paper comparison、evaluation、tests
低：citation graph、BibTeX export、Docker
```

