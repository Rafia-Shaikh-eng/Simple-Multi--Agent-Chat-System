
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from helpers import Message, pretty
from memory import MemoryStore
from agents.agents import ResearchAgent, AnalysisAgent, MemoryAgent, AgentResult

@dataclass
class TraceEntry:
    agent: str
    input: str
    output: str
    confidence: float
    meta: Dict[str, Any]

class Coordinator:
    def __init__(self, knowledge_base: Dict[str, List[str]]):
        self.memory = MemoryStore()
        self.research = ResearchAgent(self.memory, knowledge_base)
        self.analysis = AnalysisAgent(self.memory)
        self.mem_agent = MemoryAgent(self.memory)
        self.conversation: List[Dict[str, Any]] = []
        self.trace: List[TraceEntry] = []

    def _log(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        self.conversation.append({"role": role, "content": content, "meta": meta or {}})

    def _complexity(self, query: str) -> str:
        # Simple heuristic: if query mentions "analyze", "compare", "trade-off", or multiple commas -> complex
        ql = query.lower()
        if any(k in ql for k in ["analyze", "compare", "trade-off", "tradeoffs", "identify", "recommend"]) or query.count(",") >= 1:
            return "complex"
        if any(k in ql for k in ["memory", "earlier", "previous", "what did we"]):
            return "memory"
        return "simple"

    def ask(self, user_query: str) -> Dict[str, Any]:
        self._log("user", user_query)
        mode = self._complexity(user_query)
        plan = []

        # Planner with memory reuse
        mem_hits = self.memory.search(user_query, top_k=2)
        if mem_hits:
            # If good hit, prioritize memory agent
            plan.append(("memory", f"{user_query}"))
        if mode == "simple":
            plan.append(("research", user_query))
        elif mode == "complex":
            plan.extend([("research", user_query), ("analysis", "analyze findings")])
        else:  # memory mode explicit
            plan.append(("memory", user_query))

        # Execute plan
        context: Dict[str, Any] = {}
        final_answer = ""
        for agent_name, task in plan:
            if agent_name == "research":
                res = self.research.run(task, context)
                self.trace.append(TraceEntry(agent="research", input=task, output=res.content, confidence=res.confidence, meta=res.trace))
                context["research_output"] = res.content
                final_answer = res.content
            elif agent_name == "analysis":
                res = self.analysis.run(task, context)
                self.trace.append(TraceEntry(agent="analysis", input=task, output=res.content, confidence=res.confidence, meta=res.trace))
                final_answer = res.content
            elif agent_name == "memory":
                res = self.mem_agent.run(task, context)
                self.trace.append(TraceEntry(agent="memory", input=task, output=res.content, confidence=res.confidence, meta=res.trace))
                final_answer = res.content
            else:
                final_answer = "Unknown agent."

        # Save conversation message
        self.memory.add(
            kind="conversation",
            topic=user_query,
            content=final_answer,
            source="conversation",
            agent="coordinator",
            confidence=0.8,
            metadata={"plan": plan}
        )
        self._log("system", final_answer)
        return {
            "answer": final_answer,
            "mode": mode,
            "plan": plan,
            "trace": [t.__dict__ for t in self.trace],
        }
