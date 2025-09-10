
from coordinator import Coordinator
from knowledge_base import KB

def run_query(q: str):
    coord = Coordinator(KB)
    result = coord.ask(q)
    print("ANSWER:\n", result["answer"])
    print("\nMODE:", result["mode"])
    print("PLAN:", result["plan"])
    print("\nTRACE:")
    for t in result["trace"]:
        print(f"- {t['agent']} (conf={t['confidence']:.2f}): {t['meta']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Enter your question: ")
    run_query(q)
