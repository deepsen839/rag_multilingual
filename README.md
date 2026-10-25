# Multilingual CPU RAG Chatbot

A multilingual, CPU-only Retrieval-Augmented Generation (RAG) chatbot for querying large private PDF collections.

## Features

- Multiple PDF upload
- PyMuPDF native PDF text extraction
- Page-level OCR fallback for scanned/low-text pages
- PaddleOCR multilingual OCR
- Unicode normalization and text cleaning
- Repeated header/footer removal
- Language detection
- Configurable chunking with overlap
- Multilingual sentence-transformer embeddings
- Qdrant vector database
- HNSW approximate nearest-neighbor search
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF) hybrid retrieval
- Groq LLM answer generation
- PDF filename/page provenance
- Streamlit UI
- Docker Compose deployment
- Incremental ingestion using SHA-256 fingerprints
- Retrieval and end-to-end latency percentile reporting

## Architecture

```text
PDFs
  |
  v
PyMuPDF native extraction
  |
  +-- sufficient text --> cleaning
  |
  +-- insufficient text --> PaddleOCR --> cleaning
                                  |
                                  v
                         language detection
                                  |
                                  v
                           chunking
                                  |
                                  v
                     multilingual embeddings
                           /                                     v             v
                      Qdrant          BM25
                       HNSW             |
                          \            /
                           v          v
                            RRF fusion
                                 |
                                 v
                             Top-K chunks
                                 |
                                 v
                              Groq LLM
                                 |
                                 v
                         Answer + citations
```

## Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF processing | PyMuPDF |
| OCR | PaddleOCR |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector database | Qdrant |
| ANN index | HNSW |
| Lexical retrieval | BM25 |
| Hybrid fusion | Reciprocal Rank Fusion |
| LLM | Groq |
| Runtime | Python 3.11 |
| Deployment | Docker / Docker Compose |
| Target | CPU-only |

## Project Structure

```text
rag_multilingual_cpu/
├── data/
│   ├── pdfs/
│   └── processed/
├── storage/
│   ├── bm25.pkl
│   ├── documents.json
│   └── metrics.jsonl
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── pdf_processor.py
│   ├── paddle_ocr.py
│   ├── ocr.py
│   ├── language.py
│   ├── text_cleaning.py
│   ├── chunker.py
│   ├── qdrant_store.py
│   ├── bm25_store.py
│   └── retriever.py
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

## Requirements

Recommended hardware:

- 64-bit CPU
- 16 GB RAM or more for large OCR workloads
- SSD storage
- Docker and Docker Compose

OCR is the most memory-intensive stage. Large scanned pages should be rendered at controlled resolution.

## Configuration

Create `.env`:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=pdf_chunks

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
BATCH_SIZE=16

CHUNK_SIZE=400
CHUNK_OVERLAP=60

OCR_DPI=120
MAX_OCR_DIMENSION=1600
MIN_NATIVE_CHARS=50

DENSE_TOP_K=10
BM25_TOP_K=10
FINAL_TOP_K=5
RRF_K=60

GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=llama-3.3-70b-versatile
```

Never commit a real API key. Add `.env` to `.gitignore`.

## Docker Compose

The application uses Streamlit and Qdrant:

```yaml
services:
  rag-streamlit:
    build: .
    container_name: rag-streamlit
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - qdrant
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
      - hf_cache_volume:/root/.cache/huggingface
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_storage:
  hf_cache_volume:
```

Inside Docker:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

Do not use `localhost` for Qdrant from the Streamlit container.

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd rag_multilingual_cpu
cp .env.example .env
```

Edit `.env`, then:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Open:

```text
http://localhost:8501
```

## PDF Ingestion

Place PDFs in:

```text
data/pdfs/
```

Run:

```bash
docker compose exec rag-streamlit python -m src.ingest
```

The pipeline performs:

1. PDF discovery
2. SHA-256 fingerprinting
3. Native text extraction
4. OCR fallback for insufficient native text
5. Unicode normalization
6. Repeated header/footer removal
7. Language detection
8. Chunk creation
9. Embedding generation
10. Qdrant upsert
11. BM25 indexing
12. Registry update

Unchanged documents are skipped. Changed documents are rebuilt.

## OCR

The system first attempts native PDF extraction. OCR is used only when native text is below the configured threshold.

```text
PDF page
   |
PyMuPDF
   |
enough text?
  / yes  no
 |    |
 | PaddleOCR
  \  /
   \/
cleaned text
```

For CPU-only systems:

```env
OCR_DPI=120
MAX_OCR_DIMENSION=1600
```

Check the installed PaddleOCR version:

```bash
docker compose exec rag-streamlit python -c "import paddleocr; print(paddleocr.__version__)"
```

## Retrieval

Dense retrieval uses multilingual embeddings and Qdrant HNSW.

Typical HNSW settings:

```text
m = 32
ef_construct = 200
ef_search = 64
distance = COSINE
```

BM25 handles lexical matching. Dense and BM25 results are merged with Reciprocal Rank Fusion:

```text
RRF_score(d) = sum(1 / (k + rank(d)))
```

with:

```env
RRF_K=60
```

## Query Flow

```text
Question
   |
   +--> Qdrant dense retrieval
   |
   +--> BM25 retrieval
   |
   v
RRF fusion
   |
Top-K chunks
   |
Groq LLM
   |
Answer + PDF/page citations
```

## Citations

Retrieved chunks retain:

- document ID
- filename
- page number
- language
- source type
- chunk ID
- text

The answer should cite sources using filename and page, for example:

```text
Sources:
- Let Us C, page 29
- Let Us C, page 30
```

The generation prompt should instruct the model to answer only from retrieved context and avoid unsupported claims.

## Performance

Track:

- P50 retrieval latency
- P90 retrieval latency
- P95 retrieval latency
- P99 retrieval latency
- P50 end-to-end latency
- P90 end-to-end latency
- P95 end-to-end latency
- P99 end-to-end latency

Separate offline ingestion latency from online query latency.

The 2–5 second target should be validated with benchmarks on the actual CPU, RAM, corpus, and LLM configuration. Cold model startup can increase latency.

## Evaluation

Recommended retrieval metrics:

- Recall@k
- MRR
- Precision@k

System metrics:

- P50/P95/P99 latency
- Citation accuracy
- Hallucination rate

Formal retrieval metrics require a ground-truth evaluation set containing relevant documents/pages/chunks.

## Monitoring

Application logs:

```bash
docker compose logs --tail=100 rag-streamlit
```

Follow logs:

```bash
docker compose logs --tail=100 -f rag-streamlit
```

Resource usage:

```bash
docker stats rag-streamlit
```

Container state:

```bash
docker inspect rag-streamlit   --format='Status={{.State.Status}} OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} RestartCount={{.RestartCount}}'
```

Processes:

```bash
docker top rag-streamlit
```

If `OOMKilled=true`, the container was terminated because of memory exhaustion.

## Restarting

Normal restart:

```bash
docker compose restart rag-streamlit
```

Rebuild:

```bash
docker compose up -d --build rag-streamlit
```

Stop without removing volumes:

```bash
docker compose down
```

Avoid:

```bash
docker compose down -v
```

unless you intentionally want to delete persistent Qdrant and model-cache volumes.

## Troubleshooting

### OCR is extremely slow

Check:

```bash
docker stats rag-streamlit
```

Use:

```env
OCR_DPI=120
MAX_OCR_DIMENSION=1600
```

A large scanned corpus can make OCR the dominant ingestion cost.

### Container is OOM-killed

Check:

```bash
docker inspect rag-streamlit   --format='OOMKilled={{.State.OOMKilled}} RestartCount={{.RestartCount}}'
```

Reduce OCR image dimensions and use a lighter OCR model where supported.

### Qdrant connection error

Check:

```bash
docker compose ps
docker compose logs --tail=100 qdrant
```

The application container should use:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### Streamlit appears stuck

Check:

```bash
docker compose logs --tail=200 rag-streamlit
docker stats rag-streamlit --no-stream
docker top rag-streamlit
```

If there is no ingestion process and CPU usage is low, the ingestion job may have stopped or the container may have restarted.

## Security

Do not commit:

```text
.env
*.key
*.pem
credentials*
secrets*
```

If an API key is exposed, revoke/rotate it immediately.

Private PDF documents should not be committed to a public repository without appropriate rights.

## Reproducibility

Use:

- pinned dependencies
- deterministic chunking
- explicit embedding model versions
- explicit retrieval parameters
- deterministic document/chunk IDs
- persisted processed documents
- persisted Qdrant storage
- persisted BM25 index
- recorded benchmark queries and metrics

## Design Rationale

### PyMuPDF

Fast native PDF text extraction and page-level access make it suitable for routing pages between native extraction and OCR.

### PaddleOCR

Provides OCR for scanned pages where a usable text layer is unavailable.

### Multilingual embeddings

Allow queries and passages from multiple languages to be represented in a shared semantic space.

### Qdrant + HNSW

Provides persistent vector storage and approximate nearest-neighbor retrieval.

### BM25 + dense retrieval

Dense retrieval captures semantic similarity while BM25 handles exact lexical terms, names, numbers, identifiers, and code.

### RRF

RRF combines ranked results without requiring dense and BM25 scores to be directly comparable.

## Limitations

- OCR is CPU- and memory-intensive.
- OCR quality depends on scan quality, resolution, language, and layout.
- Complex tables and diagrams may require specialized document understanding.
- Query latency depends on hardware, model warm-up, corpus size, and LLM API latency.
- Formal evaluation requires labeled ground truth.
- Large scanned PDFs may require batch ingestion and resource controls.

## Future Improvements

- Background ingestion workers
- Page-level ingestion checkpoints
- OCR retries and timeouts
- Tokenizer-aware chunking
- Query/result caching
- Cross-encoder reranking
- Access control
- Structured evaluation datasets
- Monitoring with Prometheus/Grafana
- Rate limiting
- Separate ingestion and query services

## License

Add the appropriate license for the source code.

Do not redistribute copyrighted PDF documents unless you have the required rights or permission.
