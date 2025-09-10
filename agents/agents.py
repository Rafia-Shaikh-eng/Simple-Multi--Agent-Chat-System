
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from helpers import Message, pretty
from memory import MemoryStore

@dataclass
class AgentResult:
    content: str
    confidence: float
    trace: Dict[str, Any]

class BaseAgent:
    name: str = "base"
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def run(self, task: str, context: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError

class ResearchAgent(BaseAgent):
    name = "research"
    def __init__(self, memory: MemoryStore, knowledge_base: Dict[str, List[str]]):
        super().__init__(memory)
        self.kb = knowledge_base

    def run(self, task: str, context: Dict[str, Any]) -> AgentResult:
        # Simulate a "web search" over a curated KB (dict of topic -> facts)
        hits: List[str] = []
        q = task.lower()
        for topic, facts in self.kb.items():
            if any(tok in topic.lower() for tok in q.split()) or any(
                tok in " ".join(facts).lower() for tok in q.split()
            ):
                hits.extend(facts)
        if not hits:
            hits = ["No direct facts found; consider broadening the query."]
            conf = 0.4
        else:
            conf = 0.75 if len(hits) <= 5 else 0.65
        content = "\n".join(f"- {h}" for h in hits)
        # Save to knowledge memory
        self.memory.add(
            kind="knowledge",
            topic=task,
            content=content,
            source="mock_kb",
            agent=self.name,
            confidence=conf,
            metadata={"facts_count": len(hits)}
        )
        return AgentResult(content=content, confidence=conf, trace={"task": task, "hits": len(hits)})

class AnalysisAgent(BaseAgent):
    name = "analysis"
    def run(self, task: str, context: Dict[str, Any]) -> AgentResult:
        # Basic comparative analysis with simple heuristics
        inputs = context.get("research_output", "")
        lines = [l.strip("- ").strip() for l in inputs.splitlines() if l.strip()]
        # Simple scoring by presence of keywords
        scores = {}
        for l in lines:
            s = 1
            for kw, w in [
                ("efficient", 2), ("converge", 2), ("robust", 2),
                ("scalable", 2), ("memory", 1), ("compute", 1),
                ("simple", 1), ("state-of-the-art", 3)
            ]:
                if kw in l.lower():
                    s += w
            # Token-based heuristic
            s += len(l.split()) / 20.0
            scores[l] = s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        summary = []
        for i, (line, sc) in enumerate(ranked[:5], 1):
            summary.append(f"{i}. {line} (score={sc:.2f})")
        rec = ranked[0][0] if ranked else "Insufficient data for recommendation."
        conf = 0.6 + min(0.3, len(ranked)/20.0)
        content = "Top findings:\n" + "\n".join(summary) + f"\n\nRecommendation: {rec}"
        # Save agent state
        self.memory.add(
            kind="agent_state",
            topic="analysis_result",
            content=content,
            source="analysis",
            agent=self.name,
            confidence=conf,
            metadata={"based_on": task}
        )
        return AgentResult(content=content, confidence=conf, trace={"scored_items": len(ranked)})

class MemoryAgent(BaseAgent):
    name = "memory"
    def run(self, task: str, context: Dict[str, Any]) -> AgentResult:
        # task can be "search topic: X"
        q = task
        hits = self.memory.search(q, top_k=5)
        if not hits:
            content = "No relevant memory found."
            conf = 0.3
        else:
            lines = []
            for r in hits:
                lines.append(f"[{r.kind}] {r.topic} :: {r.content[:120].strip()}... (by {r.agent}, conf={r.confidence:.2f})")
            content = "\n".join(lines)
            conf = 0.7
        return AgentResult(content=content, confidence=conf, trace={"query": q, "results": len(hits)})
