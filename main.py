import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from kmrl_agent.agent.assistant import KMRLAssistant

def main():
    print("=" * 70)
    print("  Kochi Metro (KMRL) Agentic RAG Assistant - CLI Entrypoint")
    print("=" * 70)

    assistant = KMRLAssistant()

    test_queries = [
        "How much is ticket fare from Aluva to Pettah?",
        "What are the stations between Edapally and MG Road?",
        "Did anyone find an umbrella at Aluva station?",
        "What are the parking charges for two wheelers and cars?",
        "What is the feeder bus schedule at Kalamassery metro station?"
    ]

    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        res = assistant.process_query(q)
        print(f"[TOOL SELECTED]: {res['tool_selected']} ({res['reasoning']})")
        print(f"[ANSWER]:\n{res['answer']}\n")
        print("-" * 70)

if __name__ == "__main__":
    main()
