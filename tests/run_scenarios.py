import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coordinator import Coordinator
from knowledge_base import KB
from pathlib import Path

SCENARIOS = [
    ("simple_query.txt", "What are the main types of neural networks?"),
    ("complex_query.txt", "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."),
    ("memory_test.txt", "What did we discuss about neural networks earlier?"),
    ("multi_step.txt", "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."),
    ("collaborative.txt", "Compare two machine-learning approaches and recommend which is better for our use case.")
]

def run_all(out_dir: Path):
    coord = Coordinator(KB)
    outs = []
    for fname, prompt in SCENARIOS:
        result = coord.ask(prompt)
        text = []
        text.append(f"PROMPT: {prompt}\n")
        text.append("ANSWER:\n" + result["answer"] + "\n")
        text.append("MODE: " + result["mode"] + "\n")
        text.append("PLAN: " + str(result["plan"]) + "\n")
        text.append("TRACE:\n")
        for t in result["trace"]:
            text.append(f"- {t['agent']} (conf={t['confidence']:.2f}) meta={t['meta']}")
        data = "\n".join(text)
        (out_dir / fname).write_text(data)
        outs.append((fname, len(result['trace'])))
    return outs

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    res = run_all(out_dir)
    for f, n in res:
        print(f"Generated {f} with {n} trace entries.")
