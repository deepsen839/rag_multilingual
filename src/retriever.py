import time
from collections import defaultdict

from .qdrant_store import QdrantStore
from .bm25_store import BM25Store
from .config import (
    BM25_PATH,
    DENSE_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    RRF_K,
)


class HybridRetriever:

    def __init__(self):
        start = time.perf_counter()

        self.qdrant = QdrantStore()

        qdrant_init = time.perf_counter()

        self.bm25 = BM25Store(BM25_PATH)

        if BM25_PATH.exists():
            self.bm25.load()

        bm25_init = time.perf_counter()

        print(
            "\n========== RETRIEVER INIT =========="
        )
        print(
            f"Qdrant init : "
            f"{qdrant_init - start:.3f}s"
        )
        print(
            f"BM25 load   : "
            f"{bm25_init - qdrant_init:.3f}s"
        )
        print(
            f"TOTAL init  : "
            f"{bm25_init - start:.3f}s"
        )
        print(
            "====================================\n"
        )

    def search(self, query, top_k=FINAL_TOP_K):

        total_start = time.perf_counter()

        # -------------------------
        # QDRANT
        # -------------------------
        start = time.perf_counter()

        dense = self.qdrant.search(
            query,
            DENSE_TOP_K
        )

        dense_time = time.perf_counter() - start

        # -------------------------
        # BM25
        # -------------------------
        start = time.perf_counter()

        lexical = self.bm25.search(
            query,
            BM25_TOP_K
        )

        bm25_time = time.perf_counter() - start

        # -------------------------
        # RRF
        # -------------------------
        start = time.perf_counter()

        scores = defaultdict(float)
        payloads = {}

        for results in (dense, lexical):

            for rank, item in enumerate(results, 1):

                cid = item["chunk_id"]

                scores[cid] += (
                    1.0 / (RRF_K + rank)
                )

                payloads[cid] = item["payload"]

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_results = [
            {
                "chunk_id": cid,
                "score": score,
                "payload": payloads[cid],
            }
            for cid, score in ranked[:top_k]
        ]

        rrf_time = time.perf_counter() - start

        total_time = time.perf_counter() - total_start

        print("\n========== RETRIEVAL ==========")
        print(
            f"Qdrant : {dense_time:.3f}s"
        )
        print(
            f"BM25   : {bm25_time:.3f}s"
        )
        print(
            f"RRF    : {rrf_time:.3f}s"
        )
        print(
            f"TOTAL  : {total_time:.3f}s"
        )
        print(
            "===============================\n"
        )

        return final_results