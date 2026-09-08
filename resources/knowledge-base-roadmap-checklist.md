# Knowledge Base Roadmap — Checklist

Actionable checklist derived from the [Knowledge Base Roadmap](./knowledge-base-roadmap.html)
artifact (also mirrored at
https://claude.ai/code/artifact/383aecc4-36e4-4906-841a-17c553fd2271):
a phased plan for turning slide decks and other files into a working
RAG (retrieval-augmented generation) chatbot.

## Route decision

- [ ] Choose Route A (managed knowledge base: Amazon Bedrock / Azure AI Search / Google Vertex AI Search) **or**
      Route B (self-built pipeline: Unstructured or Docling + Qdrant or pgvector + LangChain or LlamaIndex)

## Phase 0 — Define scope, audience, and success

- [ ] Write 15–20 real questions people will actually ask (becomes the phase 6 test set)
- [ ] Decide who can see what; plan permission boundaries if files have different access levels
- [ ] Set a freshness bar (e.g. "reflects files updated within 24h" vs. "refreshed monthly")

## Phase 1 — Inventory and stage the source files

- [ ] Consolidate all source files into one place the ingestion pipeline can read continuously
- [ ] Build a file manifest (path, owner, last-modified date, sensitivity tag)
- [ ] Deduplicate and retire old/duplicate draft versions

## Phase 2 — Extract text and structure from every file type

- [ ] Pull slide text + speaker notes + tables, not just visible text boxes
- [ ] Run OCR on image-only slides and scanned PDFs
- [ ] Preserve structure while extracting (titles, section headers, table boundaries)
- [ ] Capture per-file metadata (filename, author, date, source path) alongside the text

## Phase 3 — Chunk the content

- [ ] Default to ~400–512 tokens per chunk with 10–20% overlap, split on paragraph/sentence breaks
- [ ] Chunk per slide for PPTs rather than merging unrelated slides
- [ ] Keep tables intact as their own chunk
- [ ] Attach source metadata (file name, slide/page number, date) to every chunk

## Phase 4 — Generate embeddings and build the index

- [ ] Pick one embedding model and keep it consistent across the whole index
- [ ] Choose a vector database sized for growth (see comparison table in the roadmap)
- [ ] Store metadata alongside vectors to support filtering (department, date, sensitivity)

## Phase 5 — Build retrieval + the answer pipeline

- [ ] Add hybrid search (vector similarity + keyword search)
- [ ] Add a reranking step for larger libraries
- [ ] Instruct the model to cite sources and refuse to guess when nothing relevant was retrieved

## Phase 6 — Evaluate before you trust it

- [ ] Run the phase 0 question set against the system and grade the answers
- [ ] Check faithfulness (answer matches the source documents, no fabricated detail)
- [ ] Check retrieval separately from generation (fix chunking/indexing before touching prompts)
- [ ] Track a small eval set over time to catch regressions

## Phase 7 — Ship the chat interface

- [ ] Put it where people already work (Slack/Teams bot or internal web chat)
- [ ] Show sources with every answer (link back to the exact slide/document)
- [ ] Add a feedback control (thumbs up/down)

## Phase 8 — Keep it current (ongoing)

- [ ] Automate re-ingestion on file change (watch storage, re-run phases 2–4 incrementally)
- [ ] Version and expire outdated chunks when a file is replaced
- [ ] Review the feedback queue weekly to find content gaps or chunking problems

## Reference tables (see the full artifact for details)

- [ ] Parsing tool selected (Unstructured / Docling / Apache Tika / LlamaParse / python-pptx-docx)
- [ ] Vector database selected (Pinecone / Qdrant / Weaviate / pgvector / Chroma / Milvus)
- [ ] Orchestration layer or managed platform selected (LangChain / LlamaIndex / Bedrock / Azure AI Search / Vertex AI Search)
