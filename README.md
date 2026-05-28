# psail1-basant — RAG Assistant POC

> **AI Level 1 Training Project** · by Basant Mishra

A Proof-of-Concept for a **Retrieval-Augmented Generation (RAG)** assistant that demonstrates how Large Language Models can be enhanced with private, external knowledge to produce accurate, grounded answers.

---

## Project Overview

Standard LLMs answer from training data alone, which can lead to hallucinations or gaps on private/recent topics. This POC solves that by:

1. Storing documents as semantic vector embeddings in **ChromaDB**
2. Retrieving only the most relevant chunks when a question is asked
3. Passing those chunks as context to **Qwen3-Coder 480b** (via Ollama Cloud)

The result is an assistant that answers *only* from what it actually knows.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Qwen3-Coder 480b-cloud (via Ollama Cloud API) |
| **Vector Database** | ChromaDB (persisted to `db_chroma/`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, zero cost) |
| **Orchestration** | LangChain |
| **Backend** | Python · Flask |
| **Frontend** | HTML5 · CSS3 · Vanilla JavaScript |

---

## RAG Pipeline

```
data.txt / uploaded file
        │
        ▼
  [1] Ingestion          ← TextLoader / PyPDFLoader
        │
        ▼
  [2] Chunking           ← RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
        │
        ▼
  [3] Embedding          ← HuggingFace all-MiniLM-L6-v2
        │
        ▼
  [4] Vector Storage     ← ChromaDB (persist_directory=db_chroma/)
        │
   user question
        │
        ▼
  [5] Retrieval          ← Similarity search in ChromaDB
        │
        ▼
  [6] Prompt Augmentation ← System prompt + retrieved context
        │
        ▼
  [7] Generation         ← Qwen3-Coder 480b via Ollama Cloud
        │
        ▼
     Answer shown in chat UI
```

---

## Quick Start

### Prerequisites
- Python 3.10+ (tested on 3.14)
- Internet connection (for Ollama Cloud API and HuggingFace model download)

### Install & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install faiss-cpu

# 2. Start the app
python app.py
# Or use the one-click launcher:
start_app.cmd
```

Visit **[http://localhost:5001](http://localhost:5001)** in your browser.

---

## Configuration

Edit `app.py` to change:

```python
API_KEY    = "your-ollama-cloud-key"
MODEL_NAME = "qwen3-coder:480b-cloud"   # any Ollama Cloud model
```

Default port is `5001`. Change `port=5001` in `app.run(...)` if needed.

---

## Project Structure

```
psail1-basant/
├── app.py                  # Flask backend + RAG pipeline
├── data.txt                # Default knowledge base (project overview)
├── requirements.txt        # Python dependencies
├── start_app.cmd           # One-click Windows launcher
├── README.md               # This file
├── README.html             # Rendered project page (served at /readme)
├── templates/
│   └── index.html          # Chat UI (dark theme)
├── uploads/                # Uploaded documents (auto-created)
└── db_chroma/              # ChromaDB vector store (auto-created)
```

---

## Features

- **Chat interface** — Ask questions, get answers grounded in your documents
- **Document upload** — Add PDFs or TXT files to expand the knowledge base at runtime
- **Persistent vector store** — ChromaDB saves embeddings to disk (`db_chroma/`)
- **Local embeddings** — No embedding API costs; MiniLM-L6-v2 runs entirely offline
- **Source-grounded answers** — The LLM is instructed to answer *only* from retrieved context

---

## Advantages of RAG (from project spec)

| Advantage | Description |
|---|---|
| **Accuracy** | Reduces hallucinations by grounding the model in specific data |
| **Up-to-date** | Answers questions about private or recent data the base model wasn't trained on |
| **Cost-Effective** | No need to fine-tune a massive model — just update the vector database |

---

*© 2026 Basant Mishra · AI Level 1 Training Project*
