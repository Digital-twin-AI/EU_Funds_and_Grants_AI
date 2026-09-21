import chromadb
import os
import logging

from ai_core.vector_store.bm25_index import (
    BM25Index,
    reciprocal_rank_fusion,
)

DEFAULT_CHROMA_COLLECTION = "eu_grants"

# Postavke za logiranje da vidimo šta se dešava
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBClient:
    def __init__(self):
        # Definišemo gdje će baza živjeti na disku (env-konfigurabilno;
        # default zadržava stari path da postojeći deploy nastavi raditi)
        self.db_path = os.getenv(
            "CHROMA_DB_PATH",
            os.path.join(os.getcwd(), "vector_db", "chroma_db_data"),
        )
        
        # Kreiramo folder ako ne postoji
        os.makedirs(self.db_path, exist_ok=True)
        
        # Povezujemo se na Persistent Client (to znači da podaci ostaju i kad ugasiš skriptu)
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Kreiramo kolekciju 'eu_grants'. 
            # ChromaDB automatski prepoznaje dimenziju vektora (3072) kod prvog upisa.
            self.collection = self.client.get_or_create_collection(name=DEFAULT_CHROMA_COLLECTION)
            print(f"--- ChromaDB Path: {self.db_path} ---")
            print("✅ Kolekcija 'eu_grants' spremna.")
            self._bm25 = BM25Index()
            self._rebuild_bm25_from_collection()
            
        except Exception as e:
            print(f"❌ Greška pri kreiranju Chroma klijenta: {e}")
            raise e

    def add_documents(self, documents, metadatas, ids, embeddings):
        """
        Ova funkcija je falila! Ona dodaje podatke u bazu.
        """
        try:
            # ChromaDB native metoda se zove 'add'
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            print(f"❌ Greška pri upisu u ChromaDB: {e}")
            return False

    def sync_documents(
        self,
        documents,
        metadatas,
        ids,
        embeddings,
    ):
        """Upsert a validated dataset, then remove only stale records."""
        lengths = {
            len(documents),
            len(metadatas),
            len(ids),
            len(embeddings),
        }

        if not ids:
            raise ValueError(
                "At least one document is required for synchronization."
            )

        if len(lengths) != 1:
            raise ValueError(
                "Documents, metadatas, ids and embeddings "
                "must have equal lengths."
            )

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Document IDs must be unique."
            )

        existing_ids = set(
            self.collection.get()["ids"]
        )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        stale_ids = sorted(
            existing_ids - set(ids)
        )

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        self._rebuild_bm25_from_collection()
        return self.collection.count()


    def _rebuild_bm25_from_collection(self) -> None:
        """Rebuild BM25 from all stored documents. Safe on empty collection."""
        try:
            data = self.collection.get(include=["documents"])
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            if ids and docs and len(ids) == len(docs):
                self._bm25.build(ids, docs)
                logger.info("BM25 index rebuilt: %s docs", len(ids))
            else:
                self._bm25 = BM25Index()
        except Exception as e:
            logger.warning("BM25 rebuild skipped: %s", e)
            self._bm25 = BM25Index()

    def query_hybrid(
        self,
        query_text: str,
        query_embeddings,
        n_results: int = 5,
        *,
        candidate_n: int | None = None,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """Hybrid: dense ANN + BM25, fused with RRF."""
        doc_count = self.collection.count()
        if doc_count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        n_results = max(1, min(n_results, doc_count))
        pool = candidate_n or min(max(n_results * 3, n_results), doc_count)

        vec = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = list((vec or {}).get("ids", [[]])[0] or [])

        if getattr(self, "_bm25", None) is None or self._bm25.n_docs == 0:
            self._rebuild_bm25_from_collection()

        bm25_hits = self._bm25.top_n(query_text or "", pool)
        bm25_ids = [doc_id for doc_id, _ in bm25_hits]

        fused = reciprocal_rank_fusion(
            [vec_ids, bm25_ids],
            k=rrf_k,
            weights=[vector_weight, bm25_weight],
        )
        top_ids = [doc_id for doc_id, _ in fused[:n_results]]
        rrf_scores = {doc_id: score for doc_id, score in fused}

        if not top_ids:
            return vec

        fetched = self.collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )
        by_id = {
            i: (d, m)
            for i, d, m in zip(
                fetched.get("ids") or [],
                fetched.get("documents") or [],
                fetched.get("metadatas") or [],
            )
        }

        ordered_ids, ordered_docs, ordered_metas, ordered_dists = [], [], [], []
        for doc_id in top_ids:
            if doc_id not in by_id:
                continue
            doc, meta = by_id[doc_id]
            ordered_ids.append(doc_id)
            ordered_docs.append(doc)
            ordered_metas.append(meta)
            ordered_dists.append(1.0 / (rrf_scores.get(doc_id, 1e-9)))

        return {
            "ids": [ordered_ids],
            "documents": [ordered_docs],
            "metadatas": [ordered_metas],
            "distances": [ordered_dists],
        }

    def query(self, query_embeddings, n_results=5):
        """
        Pretražuje bazu koristeći vektore.
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Greška pri pretrazi: {e}")
            return None
