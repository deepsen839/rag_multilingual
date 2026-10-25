import pickle
import re
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, path):
        self.path = path
        self.bm25 = None
        self.chunks = []

    @staticmethod
    def tokenize(text):
        return re.findall(r'\w+', text.lower(), flags=re.UNICODE)

    def build(self, chunks):
        self.chunks = chunks
        corpus = [self.tokenize(x['text']) for x in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.save()

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)
        self.bm25, self.chunks = data['bm25'], data['chunks']

    def search(self, query, limit=20):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self.tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:limit]
        return [{'chunk_id': self.chunks[i]['chunk_id'], 'score': float(s), 'payload': self.chunks[i]} for i, s in ranked]
