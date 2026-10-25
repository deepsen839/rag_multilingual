import time
from fastapi import FastAPI
from pydantic import BaseModel
from .retriever import HybridRetriever
from .generator import generate_answer

app = FastAPI(title='Multilingual RAG API')
retriever = HybridRetriever()

class Query(BaseModel):
    question: str
    top_k: int = 8

@app.get('/health')
def health():
    return {'status': 'ok', 'qdrant_chunks': retriever.qdrant.count()}

@app.post('/query')
def query(body: Query):
    started = time.perf_counter()
    results = retriever.search(body.question, body.top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000
    answer = generate_answer(body.question, results)
    total_ms = (time.perf_counter() - started) * 1000
    sources = [
        {'filename': r['payload']['filename'], 'page': r['payload']['page'], 'language': r['payload'].get('language','unknown')}
        for r in results
    ]
    return {'answer': answer, 'sources': sources, 'retrieval_ms': retrieval_ms, 'total_ms': total_ms, 'results': results}
