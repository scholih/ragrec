from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import networkx as nx

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# Contracts
# =============================================================================

@dataclass(frozen=True)
class DocChunk:
    chunk_id: str
    text: str
    meta: Dict[str, str]


@dataclass(frozen=True)
class Hit:
    chunk: DocChunk
    score: float
    source: str  # "bm25" | "vector" | "graph" | "hybrid" | "rerank"


# =============================================================================
# Small text utilities (fast + dependable)
# =============================================================================

_WS = re.compile(r"\s+")
_WORDS = re.compile(r"[a-z0-9]+")

def normalize_text(s: str) -> str:
    return _WS.sub(" ", s.strip())

def bm25_tokenize(s: str) -> List[str]:
    # BM25 likes lowercase tokens
    return _WORDS.findall(s.lower())

def minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if abs(mx - mn) < 1e-12:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

def cosine_sim(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    # q: (d,), X: (n,d) -> (n,)
    qn = q / (np.linalg.norm(q) + 1e-12)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ qn


# =============================================================================
# GraphRAG (lightweight): Entity Graph + Chunk linking
# =============================================================================

class EntityGraphIndex:
    """
    A lightweight GraphRAG-style index:
    - Extracts entities from each chunk (simple heuristic)
    - Builds an undirected co-occurrence graph
    - Maps entity -> chunk_ids
    - Query expansion: extract entities from query, expand neighbors, fetch linked chunks
    """

    def __init__(self) -> None:
        self.g = nx.Graph()
        self.entity_to_chunks: Dict[str, set[str]] = {}
        self.chunk_entities: Dict[str, List[str]] = {}

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """
        Simple entity extraction heuristic:
        - Keeps TitleCase words (e.g., "NeonDB", "SuccessFactors")
        - Keeps ALLCAPS tokens (e.g., "API", "HTTP")
        - Keeps error-code-like tokens (e.g., "E1127", "ERR_401", "0xA00F4244")
        - Keeps dotted identifiers (e.g., "apps.pricing.services")
        """
        # TitleCase-ish or CamelCase-ish words
        titleish = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)

        # ALLCAPS tokens and acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)

        # Error codes / hex / mixed
        codes = re.findall(r"\b(?:0x[a-fA-F0-9]{6,}|[A-Z]{1,5}[_-]?\d{2,}|E\d{3,6})\b", text)

        # Dotted paths (useful for dev docs)
        dotted = re.findall(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){2,}\b", text)

        # Normalize: lowercase for stable node IDs, keep original “feel” optional via meta later
        raw = titleish + acronyms + codes + dotted
        cleaned = []
        for e in raw:
            e = e.strip()
            if len(e) < 3:
                continue
            cleaned.append(e.lower())

        # De-dup while preserving order
        seen = set()
        out = []
        for e in cleaned:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def build(self, chunks: List[DocChunk]) -> None:
        for c in chunks:
            ents = self.extract_entities(c.text)
            self.chunk_entities[c.chunk_id] = ents

            # Map entity -> chunk_ids
            for e in ents:
                self.entity_to_chunks.setdefault(e, set()).add(c.chunk_id)

            # Co-occurrence edges
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = ents[i], ents[j]
                    if a == b:
                        continue
                    # Edge weight can be useful later
                    if self.g.has_edge(a, b):
                        self.g[a][b]["w"] += 1
                    else:
                        self.g.add_edge(a, b, w=1)

    def expand_query_entities(self, query: str, depth: int = 1, max_nodes: int = 30) -> List[str]:
        seeds = self.extract_entities(query)
        if not seeds:
            return []

        visited = set(seeds)
        frontier = list(seeds)

        for _ in range(depth):
            nxt = []
            for node in frontier:
                if node not in self.g:
                    continue
                # Sort neighbors by edge weight descending
                neigh = sorted(
                    self.g.neighbors(node),
                    key=lambda x: self.g[node][x].get("w", 1),
                    reverse=True
                )
                for nb in neigh[:10]:
                    if nb not in visited:
                        visited.add(nb)
                        nxt.append(nb)
                        if len(visited) >= max_nodes:
                            return list(visited)
            frontier = nxt

        return list(visited)

    def graph_retrieve_chunk_ids(self, query: str, depth: int = 1, per_entity_limit: int = 6) -> Dict[str, float]:
        """
        Returns chunk_id -> score using a simple scoring:
        - chunks matched by seed entities get higher score
        - chunks matched by expanded entities get lower score
        """
        expanded = self.expand_query_entities(query, depth=depth)
        if not expanded:
            return {}

        seeds = set(self.extract_entities(query))
        scores: Dict[str, float] = {}

        for e in expanded:
            chunk_ids = list(self.entity_to_chunks.get(e, []))
            if not chunk_ids:
                continue

            # Seed entities get stronger weight
            w = 1.0 if e in seeds else 0.55

            # Limit per entity to avoid graph flooding
            for cid in chunk_ids[:per_entity_limit]:
                scores[cid] = scores.get(cid, 0.0) + w

        return scores


# =============================================================================
# Tri-Modal Hybrid RAG Engine
# =============================================================================

class TriModalHybridRAG:
    """
    Production-ready pattern:
    - BM25 (exact keywords)
    - Vector Search (semantic similarity)
    - GraphRAG (entity graph expansion)
    - Merge with normalization + weights
    - Cross-encoder rerank (final precision)
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(rerank_model)

        self.chunks: List[DocChunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []

        self._embeddings: Optional[np.ndarray] = None  # (n, d)

        self.graph_index = EntityGraphIndex()
        self._chunk_by_id: Dict[str, DocChunk] = {}

    # -------------------------------------------------------------------------
    # Index
    # -------------------------------------------------------------------------

    def index(self, chunks: List[DocChunk]) -> None:
        self.chunks = [
            DocChunk(c.chunk_id, normalize_text(c.text), c.meta)
            for c in chunks
        ]
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}

        # BM25
        self._bm25_tokens = [bm25_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)

        # Vector
        texts = [c.text for c in self.chunks]
        self._embeddings = np.array(self.embedder.encode(texts, normalize_embeddings=True))

        # GraphRAG
        self.graph_index.build(self.chunks)

    # -------------------------------------------------------------------------
    # BM25
    # -------------------------------------------------------------------------

    def bm25_search(self, query: str, top_k: int = 12) -> List[Hit]:
        if not self._bm25:
            raise RuntimeError("Call index() before search.")
        q = bm25_tokenize(query)
        scores = self._bm25.get_scores(q)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [Hit(self.chunks[int(i)], float(scores[int(i)]), "bm25") for i in idxs]

    # -------------------------------------------------------------------------
    # Vector
    # -------------------------------------------------------------------------

    def vector_search(self, query: str, top_k: int = 12) -> List[Hit]:
        if self._embeddings is None:
            raise RuntimeError("Call index() before search.")
        q = np.array(self.embedder.encode([query], normalize_embeddings=True))[0]
        sims = cosine_sim(q, self._embeddings)
        idxs = np.argsort(sims)[::-1][:top_k]
        return [Hit(self.chunks[int(i)], float(sims[int(i)]), "vector") for i in idxs]

    # -------------------------------------------------------------------------
    # GraphRAG retrieval (entity graph)
    # -------------------------------------------------------------------------

    def graph_search(self, query: str, depth: int = 1, top_k: int = 12) -> List[Hit]:
        cid_to_score = self.graph_index.graph_retrieve_chunk_ids(query, depth=depth)
        if not cid_to_score:
            return []

        # Sort by graph score desc
        items = sorted(cid_to_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hits: List[Hit] = []
        for cid, sc in items:
            c = self._chunk_by_id.get(cid)
            if c:
                hits.append(Hit(c, float(sc), "graph"))
        return hits

    # -------------------------------------------------------------------------
    # Tri-modal Hybrid Merge
    # -------------------------------------------------------------------------

    def hybrid_candidates(
        self,
        query: str,
        bm25_top: int = 12,
        vec_top: int = 12,
        graph_top: int = 12,
        graph_depth: int = 1,
        merged_top: int = 18,
        w_bm25: float = 0.34,
        w_vec: float = 0.44,
        w_graph: float = 0.22,
    ) -> List[Hit]:
        bm25_hits = self.bm25_search(query, top_k=bm25_top)
        vec_hits = self.vector_search(query, top_k=vec_top)
        graph_hits = self.graph_search(query, depth=graph_depth, top_k=graph_top)

        # Normalize each channel
        bm25_norm = minmax_norm([h.score for h in bm25_hits])
        vec_norm = minmax_norm([h.score for h in vec_hits])
        graph_norm = minmax_norm([h.score for h in graph_hits])

        merged: Dict[str, Tuple[DocChunk, float]] = {}

        def add(hits: List[Hit], norms: List[float], weight: float) -> None:
            for h, ns in zip(hits, norms):
                cid = h.chunk.chunk_id
                prev = merged.get(cid, (h.chunk, 0.0))[1]
                merged[cid] = (h.chunk, prev + ns * weight)

        add(bm25_hits, bm25_norm, w_bm25)
        add(vec_hits, vec_norm, w_vec)
        add(graph_hits, graph_norm, w_graph)

        out = [Hit(chunk=c, score=s, source="hybrid") for (c, s) in merged.values()]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:merged_top]

    # -------------------------------------------------------------------------
    # Rerank (Cross-encoder)
    # -------------------------------------------------------------------------

    def rerank(self, query: str, candidates: List[Hit], top_k: int = 6) -> List[Hit]:
        pairs = [(query, h.chunk.text) for h in candidates]
        scores = self.reranker.predict(pairs)

        reranked = [Hit(h.chunk, float(s), "rerank") for h, s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    # -------------------------------------------------------------------------
    # Build LLM Context
    # -------------------------------------------------------------------------

    def build_context(self, hits: List[Hit], max_chars: int = 5000) -> str:
        parts: List[str] = []
        used = 0

        for h in hits:
            src = h.chunk.meta.get("source", "doc")
            header = f"[{src} | {h.chunk.chunk_id}]"
            block = f"{header}\n{h.chunk.text}\n"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)

        return "\n---\n".join(parts)

    # -------------------------------------------------------------------------
    # End-to-end Retrieve
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        bm25_top: int = 12,
        vec_top: int = 12,
        graph_top: int = 12,
        graph_depth: int = 1,
        merged_top: int = 18,
        rerank_top: int = 6,
    ) -> List[Hit]:
        candidates = self.hybrid_candidates(
            query=query,
            bm25_top=bm25_top,
            vec_top=vec_top,
            graph_top=graph_top,
            graph_depth=graph_depth,
            merged_top=merged_top,
        )
        return self.rerank(query, candidates, top_k=rerank_top)


# =============================================================================
# Demo: Plug your chunks here
# =============================================================================

def demo_chunks() -> List[DocChunk]:
    """
    Replace this with your real chunker output.
    This demo intentionally mixes:
    - IDs & codes (BM25 advantage)
    - paraphrases (Vector advantage)
    - entity relationships (GraphRAG advantage)
    """
    return [
        DocChunk(
            "c1",
            "Error E1127 happens when the access token expires. Refresh the token and retry the request.",
            {"source": "runbook"}
        ),
        DocChunk(
            "c2",
            "If authentication fails with HTTP 401, rotate the API key and regenerate the client secret.",
            {"source": "runbook"}
        ),
        DocChunk(
            "c3",
            "Refund policy: annual plans are eligible within 14 days of purchase. Monthly plans are non-refundable.",
            {"source": "policy"}
        ),
        DocChunk(
            "c4",
            "Subscription cancellation requires contacting Billing at least 7 days before renewal to avoid charges.",
            {"source": "policy"}
        ),
        DocChunk(
            "c5",
            "The Billing service depends on the Auth service. Auth validates tokens issued by IdentityProvider.",
            {"source": "architecture"}
        ),
        DocChunk(
            "c6",
            "IdentityProvider issues JWT tokens. Auth verifies JWT signature and claims before allowing access.",
            {"source": "architecture"}
        ),
        DocChunk(
            "c7",
            "If the user cannot log in, check IdentityProvider health, then verify Auth logs for JWT failures.",
            {"source": "support_guide"}
        ),
    ]


if __name__ == "__main__":
    rag = TriModalHybridRAG()
    rag.index(demo_chunks())

    query = "Why am I seeing E1127 and how is Auth related to IdentityProvider?"
    hits = rag.retrieve(query, graph_depth=2)

    print("\nTop reranked hits:\n")
    for h in hits:
        print(f"- {h.score:.4f} | {h.chunk.chunk_id} | {h.chunk.meta.get('source')}")

    print("\n\nLLM Context:\n")
    print(rag.build_context(hits))