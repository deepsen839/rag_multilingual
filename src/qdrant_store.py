import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, PointStruct, Filter, FieldCondition, MatchValue,SearchParams
from sentence_transformers import SentenceTransformer
from .config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL, BATCH_SIZE

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.ensure_collection()

    def ensure_collection(self):
        if not self.client.collection_exists(QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
            )

    @staticmethod
    def point_id(chunk_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def embed(self, texts):
        return self.model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def upsert_chunks(self, chunks):
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start+BATCH_SIZE]
            vectors = self.embed([x['text'] for x in batch])
            points = [PointStruct(id=self.point_id(x['chunk_id']), vector=v.tolist(), payload=x) for x, v in zip(batch, vectors)]
            self.client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

    def delete_document(self, document_id):
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key='document_id', match=MatchValue(value=document_id))]),
            wait=True,
        )

    def search(self, query, limit=20):
        q = self.embed([query])[0].tolist()
        response = self.client.query_points(collection_name=QDRANT_COLLECTION, query=q, limit=limit, with_payload=True,search_params=SearchParams(
            hnsw_ef=64
        ))
        return [{'chunk_id': p.payload['chunk_id'], 'score': float(p.score), 'payload': p.payload} for p in response.points]

    def count(self):
        return self.client.count(collection_name=QDRANT_COLLECTION, exact=True).count
