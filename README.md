\# HybridRAG: Customer Complaint Intelligence Platform



An end-to-end Hybrid Retrieval-Augmented Generation system for customer complaint analysis.



This project uses real consumer complaint data to build a complaint intelligence system with:



\- Dataset inspection and cleaning

\- TF-IDF keyword search

\- Semantic search using sentence embeddings

\- FAISS vector indexing

\- Saved semantic index for faster retrieval

\- Future hybrid retrieval combining keyword and semantic search

\- Future LLM structured complaint classification and response generation



\## Current Progress



Completed:



1\. Dataset inspection pipeline

2\. Cleaning pipeline for CFPB complaint data

3\. Keyword search engine using TF-IDF

4\. Semantic search engine using SentenceTransformers and FAISS

5\. Saved FAISS index builder



\## Project Structure



```txt

hybridrag-complaint-intelligence/

├── data/

│   ├── raw/

│   ├── processed/

│   └── indexes/

├── src/

│   ├── inspect\_dataset.py

│   ├── prepare\_dataset.py

│   ├── keyword\_search.py

│   ├── semantic\_search.py

│   ├── build\_semantic\_index.py

│   └── semantic\_search\_saved.py

├── notebooks/

├── requirements.txt

├── .gitignore

└── README.md

Dataset



The project is designed around the CFPB Consumer Complaint Database.



The dataset itself is not included in this repository because of file size. Place the downloaded file here:



data/raw/complaints.csv

How to Run



Create and activate a virtual environment:



python -m venv .venv

.venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt



Inspect the dataset:



python src/inspect\_dataset.py



Prepare the cleaned dataset:



python src/prepare\_dataset.py



Run keyword search:



python src/keyword\_search.py



Run semantic search:



python src/semantic\_search.py



Build and save semantic index:



python src/build\_semantic\_index.py



Run saved semantic search:



python src/semantic\_search\_saved.py

Next Steps

Build hybrid retrieval

Add LLM structured output

Add FastAPI backend

Add Streamlit dashboard

Add evaluation metrics

Dockerize the project



Save it.



\## Step 5: Initialize Git



Run these commands:



```powershell

git init

git status



Check that it does not show your large CSV or .venv.

