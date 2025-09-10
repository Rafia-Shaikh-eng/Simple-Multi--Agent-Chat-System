
"""
Lightweight structured memory + vector search (bag-of-words cosine).
Persists to JSON files in memory_store/.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json, math, os
from pathlib import Path
from helpers import now_ts, tokenize, pretty

STORE_DIR = Path(__file__).resolve().parent / "memory_store"

@dataclass
class MemoryRecord:
    id: str
    kind: str  # conversation | knowledge | agent_state
    topic: str
    content: str
    source: str
    agent: str
    confidence: float
    metadata: Dict[str, Any]

class VectorIndex:
    """Very small BoW TF-IDF-ish index with cosine similarity."""
    def __init__(self):
        self.docs: Dict[str, Dict[str, float]] = {}  # id -> tf
        self.df: Dict[str, int] = {}
        self.N: int = 0

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # l2 normalize
        norm = math.sqrt(sum(v*v for v in tf.values())) or 1.0
        return {k: v/norm for k, v in tf.items()}

    def add(self, doc_id: str, text: str):
        tokens = tokenize(text)
        tf = self._tf(tokens)
        self.docs[doc_id] = tf
        self.N += 1
        # update df (unique terms per doc)
        seen = set(tokens)
        for t in seen:
            self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        # Add-one smoothing
        return math.log((1 + self.N) / (1 + self.df.get(term, 0))) + 1.0

    def _vec(self, tokens: List[str]) -> Dict[str, float]:
        tf = self._tf(tokens)
        return {t: w * self._idf(t) for t, w in tf.items()}

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.docs:
            return []
        qvec = self._vec(tokenize(query))
        # Precompute doc weights with idf
        scores: List[Tuple[str, float]] = []
        for doc_id, tf in self.docs.items():
            score = 0.0
            for t, qv in qvec.items():
                if t in tf:
                    score += qv * (tf[t] * self._idf(t))
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class MemoryStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or STORE_DIR
        self.base_dir.mkdir(exist_ok=True)
        self.data_path = self.base_dir / "records.json"
        self.records: Dict[str, MemoryRecord] = {}
        self.index = VectorIndex()
        self._load()

    def _load(self):
        if self.data_path.exists():
            raw = json.loads(self.data_path.read_text())
            for rid, r in raw.items():
                mr = MemoryRecord(**r)
                self.records[rid] = mr
                self.index.add(rid, f"{mr.topic} {mr.content}")
        else:
            self._persist()

    def _persist(self):
        raw = {rid: asdict(r) for rid, r in self.records.items()}
        self.data_path.write_text(json.dumps(raw, indent=2))

    def add(self, kind: str, topic: str, content: str, source: str, agent: str,
            confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        rid = f"{kind}:{len(self.records)+1:06d}"
        mr = MemoryRecord(
            id=rid, kind=kind, topic=topic, content=content, source=source,
            agent=agent, confidence=confidence, metadata=metadata or {"ts": now_ts()}
        )
        self.records[rid] = mr
        self.index.add(rid, f"{topic} {content}")
        self._persist()
        return mr

    def search(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        # Keyword + vector hybrid: union of keyword matches with vector top_k
        # Keyword pass
        kw_hits = []
        for r in self.records.values():
            blob = f"{r.topic} {r.content}".lower()
            if all(k in blob for k in query.lower().split() if k.strip()):
                kw_hits.append(r)
        vec_ids = [rid for rid, _ in self.index.search(query, top_k=top_k)]
        vec_hits = [self.records[rid] for rid in vec_ids if rid in self.records]
        # Merge preserving order, prefer vec
        seen = set()
        merged: List[MemoryRecord] = []
        for r in vec_hits + kw_hits:
            if r.id not in seen:
                merged.append(r); seen.add(r.id)
        return merged[:top_k]

    def all(self) -> List[MemoryRecord]:
        return list(self.records.values())
