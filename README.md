# HybridRAG: Customer Complaint Intelligence

A production-inspired complaint retrieval and analysis system built on the CFPB Consumer Complaint Database. It combines lexical TF-IDF retrieval, dense semantic search with SentenceTransformers + FAISS, and Reciprocal Rank Fusion (RRF) to surface relevant historical complaints and produce retrieval-grounded product/issue predictions.

## What is implemented

- Robust CFPB dataset inspection and chunked cleaning pipeline
- Duplicate/missing-ID handling and conservative narrative normalization
- TF-IDF lexical retrieval with unigrams + bigrams
- Prebuilt dense embeddings with \`all-MiniLM-L6-v2\`
- Saved FAISS \`IndexFlatIP\` semantic index for fast query-time search
- Hybrid retrieval with Reciprocal Rank Fusion instead of mixing incompatible raw scores
- Retrieval-grounded product and issue prediction using weighted evidence voting
- FastAPI endpoints for health, search, and analysis
- Streamlit dashboard with evidence inspection and dataset overview
- Proxy retrieval evaluation using CFPB product/issue labels
- Unit tests, GitHub Actions CI, Dockerfile, and Docker Compose

## Architecture

\`\`\`text
CFPB complaints.csv
        |
        v
inspect_dataset.py
        |
        v
prepare_dataset.py  -->  data/processed/complaints_clean.csv
        |                              |
        |                              +--> TF-IDF lexical index (runtime)
        v
build_semantic_index.py
        |
        +--> FAISS index
        +--> embeddings.npy
        +--> metadata.json

User complaint
      |
      +--------------------+
      |                    |
      v                    v
 TF-IDF search       Semantic FAISS search
      |                    |
      +---------+----------+
                v
      Reciprocal Rank Fusion
                |
                v
      Top evidence complaints
                |
                v
 Retrieval-grounded product/issue prediction
                |
        +-------+-------+
        |               |
     FastAPI        Streamlit
\`\`\`

## Why hybrid retrieval?

Lexical retrieval is strong at exact terms such as company names, product names, and legal phrases. Dense semantic retrieval is strong when two complaints describe the same problem using different wording. RRF combines their rankings without pretending that TF-IDF similarity and dense cosine-style similarity are directly comparable.

## Dataset

Download the CFPB Consumer Complaint Database CSV from:

https://www.consumerfinance.gov/data-research/consumer-complaints/

Place the extracted CSV at:

\`\`\`text
data/raw/complaints.csv
\`\`\`

Large source data, processed data, embeddings, and FAISS artifacts are intentionally excluded from Git.

## Setup

Python 3.11 is recommended.

### Windows PowerShell

\`\`\`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

### macOS / Linux

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Build the data and indexes

Inspect the raw dataset:

\`\`\`bash
python -m src.inspect_dataset
\`\`\`

Prepare up to 20,000 clean complaint narratives:

\`\`\`bash
python -m src.prepare_dataset
\`\`\`

Build and persist the dense index:

\`\`\`bash
python -m src.build_semantic_index
\`\`\`

The first semantic-index build downloads the embedding model and embeds the corpus, so it is intentionally an offline step. Query-time semantic search loads the saved index instead of rebuilding all embeddings.

## Run search from the terminal

Keyword search:

\`\`\`bash
python -m src.keyword_search
\`\`\`

Semantic search:

\`\`\`bash
python -m src.semantic_search
\`\`\`

Hybrid search:

\`\`\`bash
python -m src.hybrid_search
\`\`\`

## Run the API

\`\`\`bash
uvicorn src.api:app --reload
\`\`\`

Open interactive API docs at:

\`\`\`text
http://127.0.0.1:8000/docs
\`\`\`

Endpoints:

\`\`\`text
GET  /health
POST /search
POST /analyze
\`\`\`

Example \`/search\` request:

\`\`\`json
{
  "query": "Someone opened accounts using my identity and damaged my credit report.",
  "top_k": 5
}
\`\`\`

## Run the dashboard

\`\`\`bash
streamlit run src/dashboard.py
\`\`\`

The dashboard shows a retrieval-grounded product/issue prediction plus the historical complaints used as evidence, including lexical rank, semantic rank, and fused hybrid score.

## Evaluate retrieval

The evaluation script treats matching CFPB product and issue labels as a proxy for relevance and removes the query complaint itself from the result list.

\`\`\`bash
python -m src.evaluate_retrieval --sample-size 100 --top-k 5
\`\`\`

It reports product/issue Hit@K and Mean Reciprocal Rank for keyword, semantic, and hybrid retrieval. These are proxy metrics, not human relevance judgments.

## Tests

\`\`\`bash
pytest
\`\`\`

CI runs lightweight unit tests on every push and pull request.

## Docker

After the processed dataset and semantic index have been built locally:

\`\`\`bash
docker compose up --build
\`\`\`

Services:

- API: \`http://localhost:8000\`
- Streamlit: \`http://localhost:8501\`

The local \`data/\` directory is mounted into both containers.

## Configuration

Optional environment variables:

\`\`\`text
HYBRIDRAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
HYBRIDRAG_MAX_ROWS=20000
HYBRIDRAG_CHUNKSIZE=50000
HYBRIDRAG_MIN_TEXT_LENGTH=40
HYBRIDRAG_TOP_K=5
HYBRIDRAG_CANDIDATE_K=30
HYBRIDRAG_RRF_K=60
\`\`\`

## Current scope

This repository now represents a complete **hybrid retrieval MVP**. It does not pretend to have a generative LLM layer that is not configured. The \`/analyze\` endpoint is deliberately retrieval-grounded: it predicts product and issue from weighted evidence rather than inventing prose or fake confidence values.

A future generative layer can consume the retrieved evidence to produce summaries, recommended actions, or draft responses with strict structured output. That layer should be evaluated separately for factual grounding and unsupported claims.

## Project structure

\`\`\`text
.
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── api.py
│   ├── build_semantic_index.py
│   ├── config.py
│   ├── dashboard.py
│   ├── evaluate_retrieval.py
│   ├── fusion.py
│   ├── hybrid_search.py
│   ├── inspect_dataset.py
│   ├── keyword_search.py
│   ├── prepare_dataset.py
│   └── semantic_search.py
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
\`\`\`

## Notes on responsible use

This is a portfolio/research project, not a production decision system. CFPB labels are used as weak supervision/proxy relevance signals, and retrieval agreement is not a calibrated probability. Any real deployment in financial services would require stronger evaluation, privacy controls, monitoring, access control, auditability, and human review.
