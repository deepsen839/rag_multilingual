from collections import defaultdict
from .qdrant_store import QdrantStore
from .bm25_store import BM25Store
from .config import BM25_PATH, DENSE_TOP_K, BM25_TOP_K, FINAL_TOP_K, RRF_K

class HybridRetriever:
    def __init__(self):
        self.qdrant = QdrantStore()
        self.bm25 = BM25Store(BM25_PATH)
        if BM25_PATH.exists():
            self.bm25.load()

    def search(self, query, top_k=FINAL_TOP_K):
        dense = self.qdrant.search(query, DENSE_TOP_K)
        lexical = self.bm25.search(query, BM25_TOP_K)
        scores = defaultdict(float)
        payloads = {}
        for results in (dense, lexical):
            for rank, item in enumerate(results, 1):
                cid = item['chunk_id']
                scores[cid] += 1.0 / (RRF_K + rank)
                payloads[cid] = item['payload']
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'chunk_id': cid, 'score': score, 'payload': payloads[cid]} for cid, score in ranked[:top_k]]
