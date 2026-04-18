# 🧠 Memory OS

### _An Asynchronous Graph-RAG System for Personal Intelligence_

Memory OS is a sophisticated personal knowledge management system that transcends simple note-taking. It leverages a **Hybrid Knowledge Graph** architecture, combining semantic vector search with relational graph data to create a "digital twin" of your memories, emotions, and behavioral patterns.

---

## 🚀 Key Features

- **Hybrid Knowledge Graph Architecture**: Engineered a custom data pipeline that extracts structured entities, emotions, and sentiments from unstructured **text** using Gemini, storing them simultaneously as interconnected **Graph Nodes** and **High-Dimensional Vectors** in Neo4j.
- **Temporal-Emotional Re-ranker**: Solved the "stale context" flaw in standard RAG architectures with a custom retrieval algorithm. Memories are dynamically scored via cosine similarity and scaled by LLM-extracted emotional intensity and exponential time decay:
  $$Score = Similarity \cdot Emotion \cdot e^{-k \cdot t}$$
  _(where $k$ is the decay rate and $t$ is the age of the memory in days)_.
- **Asynchronous Insight Engine**: A background processing layer that aggregates weekly user data through a 1M+ token context window to extract behavioral patterns and programmatically injects high-level **Insight Nodes** back into the Knowledge Graph to enable complex meta-querying.

---

## 🏗️ Architecture

The system follows a multi-stage pipeline for recording and retrieving personal data:

1.  **Extraction**: Raw text is processed by **Gemini 3.1 Pro** via Pydantic AI to extract entities, sentiment score, emotional intensity, primary emotion, and action items as a structured schema.
2.  **Embedding**: A "rich context" string (combining the raw text, summary, entities, and emotion) is embedded into a **3072-dimensional vector** using `gemini-embedding-001`.
3.  **Storage**: The full memory package — raw text, extracted metadata, and embedding vector — is persisted in **Neo4j** simultaneously as a `Memory` graph node (with `[:MENTIONS]` relationships to `Entity` nodes) and as a searchable vector via a cosine similarity index.
4.  **Retrieval & Scoring**: Incoming queries are embedded and matched against the vector index. The top-10 semantic candidates are then re-ranked using the Temporal-Emotional scorer before the top 3 are passed to the LLM for synthesis.
5.  **Insight Generation**: A periodic job (`run_weekly_insight_job`) fetches recent memories, uses Gemini to identify behavioral patterns, and writes a structured `Insight` node back into the graph — connected to the relevant `Entity` nodes via `[:DERIVED_FROM]` relationships.

---

## 🛠️ Tech Stack

| Category              | Technology                                              |
| :-------------------- | :------------------------------------------------------ |
| **LLM Orchestration** | Pydantic AI, Gemini 3.1 Pro                             |
| **Embeddings**        | Google Generative AI (`gemini-embedding-001`, 3072-dim) |
| **Database**          | Neo4j (Graph + Cosine Vector Index)                     |
| **Logic & API**       | Python, FastAPI                                         |
| **Scoring**           | NumPy (Temporal-Emotional re-ranking)                   |
| **Environment**       | python-dotenv                                           |

---

## 📂 Project Structure

| File                | Role                                                                                        |
| :------------------ | :------------------------------------------------------------------------------------------ |
| `extraction.py`     | LLM-based structured data extraction (`MemoryData` schema via Pydantic AI)                  |
| `embeddings.py`     | Master pipeline: calls extraction, builds rich context string, generates 3072-dim vector    |
| `database.py`       | Neo4j driver wrapper; handles schema setup, vector index creation, and memory insertion     |
| `scoring.py`        | Temporal-Emotional re-ranking algorithm (`calculate_memory_score`)                          |
| `retrieval.py`      | Hybrid search (vector query + Cypher graph traversal), re-ranking, and LLM synthesis        |
| `insight_engine.py` | Weekly background job: fetches memories, extracts patterns, writes `Insight` nodes to graph |
