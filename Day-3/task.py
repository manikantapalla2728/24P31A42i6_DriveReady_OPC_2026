"""
Day 3 - Your first agent: what an agent is, and LangChain
Case study: Sweet Crumbs Home Bakery

What this file does
-------------------
* Rebuilds the bakery assistant in LangChain:  prompt | llm | parser   (a "chain")
* Adds memory with MessagesPlaceholder
* Tests the assistant from test.csv (question + must_include) and prints the pass rate
* "Your 30%": improve the prompt and compare pass rates, run at temperature 1.0 twice
* Sneak peek: the same assistant as a LangChain agent (no tools yet - tools come on Day 4)

Run
---
    pip install langchain langchain-groq
    export GROQ_API_KEY="your-key"      # Windows PowerShell: $env:GROQ_API_KEY="your-key"
    python task.py                      # tests + improve + temperature + agent
    python task.py chat | tests | improve | temperature | agent
"""

import argparse
import csv
import os
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

MODEL = "openai/gpt-oss-120b"
TESTS_FILE = Path(__file__).parent / "test.csv"


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def ensure_api_key() -> None:
    """LangChain reads GROQ_API_KEY from the environment; fall back to Colab Secrets."""
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        from google.colab import userdata

        os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
    except Exception:
        raise SystemExit("GROQ_API_KEY not found. Set it as an environment variable.")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80 each.
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time or place. If it isn't in FACTS,
        say "I'll check with the owner." Work out totals carefully.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""

# "Improve": ONE change to the prompt - an explicit rule for same-day requests.
SYSTEM_V2 = SYSTEM.replace(
    "Work out totals carefully.",
    "Work out totals carefully.\n"
    "        If a customer wants a cake the same day, say clearly that we need 24 hours' notice.",
)


# --------------------------------------------------------------------------- #
# Chain = template | model | parser
# --------------------------------------------------------------------------- #
def build_chain(system: str = SYSTEM, temperature: float = 0.3, llm=None):
    """Build the bakery chain. `llm` can be injected (useful for testing without the API)."""
    if llm is None:
        llm = ChatGroq(model=MODEL, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("history"),   # the chat so far goes here
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


class Assistant:
    """A chain plus memory: every turn is appended to `history` and sent next time."""

    def __init__(self, chain):
        self.chain = chain
        self.history: list[tuple[str, str]] = []

    def ask(self, question: str) -> str:
        answer = self.chain.invoke({"history": self.history, "question": question})
        self.history.extend([("human", question), ("ai", answer)])
        return answer


# --------------------------------------------------------------------------- #
# Tests: real customer questions + a word the answer MUST include
# --------------------------------------------------------------------------- #
def load_tests(path: Path = TESTS_FILE) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_tests(chain, path: Path = TESTS_FILE, verbose: bool = True) -> dict:
    """Ask every question with a fresh history; PASS if must_include appears in the answer."""
    rows = load_tests(path)
    results, passed = [], 0
    for row in rows:
        try:
            answer = chain.invoke({"history": [], "question": row["question"]})
        except Exception as error:  # an API failure counts as a failed test, not a crash
            answer = f"<error: {error}>"
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        results.append({"question": row["question"], "ok": ok, "answer": answer})
        if verbose:
            print("PASS" if ok else "FAIL", "|", row["question"], "->", answer[:70].replace("\n", " "))
    rate = passed / len(rows) if rows else 0.0
    if verbose:
        print(f"\n{passed}/{len(rows)} passed ({rate:.0%})")
    return {"passed": passed, "total": len(rows), "rate": rate, "results": results}


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def demo_chat() -> None:
    header("Assistant with memory")
    assistant = Assistant(build_chain())
    print(assistant.ask("Hi, I'm Ravi. Do you make eggless cakes?"))
    print(assistant.ask("What's my name?"))


def demo_tests() -> None:
    header("Tests: baseline prompt (temperature 0.3)")
    run_tests(build_chain(SYSTEM, 0.3))


def demo_improve() -> None:
    header("Improve: change ONE thing in the prompt and compare pass rates")
    print("--- v1 (original prompt) ---")
    v1 = run_tests(build_chain(SYSTEM, 0.3))
    print("\n--- v2 (added a same-day rule) ---")
    v2 = run_tests(build_chain(SYSTEM_V2, 0.3))
    change = v2["rate"] - v1["rate"]
    verdict = "went UP" if change > 0 else "went DOWN" if change < 0 else "did not change"
    print(f"\nPass rate {v1['rate']:.0%} -> {v2['rate']:.0%}: it {verdict}.")


def demo_temperature() -> None:
    header("Switch: temperature 1.0, run the tests twice")
    chain = build_chain(SYSTEM, temperature=1.0)
    runs = []
    for number in (1, 2):
        print(f"\n--- run {number} ---")
        runs.append(run_tests(chain))
    flipped = [a["question"] for a, b in zip(runs[0]["results"], runs[1]["results"]) if a["ok"] != b["ok"]]
    print(f"\nPass rates: {runs[0]['rate']:.0%} and {runs[1]['rate']:.0%}")
    print("Questions whose result changed between runs:", flipped or "none")


def demo_agent() -> None:
    header("Sneak peek: the same assistant as a LangChain agent (no tools yet)")
    try:
        from langchain.agents import create_agent
    except ImportError:
        print("create_agent needs a recent `langchain` (pip install -U langchain).")
        return
    agent = create_agent(model=ChatGroq(model=MODEL, temperature=0.3), tools=[], system_prompt=SYSTEM)
    result = agent.invoke({"messages": [{"role": "user", "content": "Do you make eggless cakes?"}]})
    print(result["messages"][-1].content)


DEMOS = {
    "chat": demo_chat,
    "tests": demo_tests,
    "improve": demo_improve,
    "temperature": demo_temperature,
    "agent": demo_agent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 3 - Sweet Crumbs assistant in LangChain")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", *DEMOS],
                        help="what to run (default: all)")
    args = parser.parse_args()

    ensure_api_key()
    selected = list(DEMOS.values()) if args.mode == "all" else [DEMOS[args.mode]]
    for demo in selected:
        demo()


if __name__ == "__main__":
    main()
