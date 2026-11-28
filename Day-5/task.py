"""
Day 5 - Memory and RAG: an agent that remembers and looks things up
Case study: Sweet Crumbs Home Bakery  (the complete single agent)

Brain + instructions + tools + memory + knowledge + tests:
  * Brain         ChatGroq, low temperature
  * Instructions  system prompt: Role / Rules / Format
  * Tools         get_price, is_open, log_order (Day 4)  +  search_faq (today, RAG)
  * Memory        checkpointer (InMemorySaver) + one thread_id per customer
  * Knowledge     ChromaDB collection of the bakery's FAQs and policies
  * Checks        Pydantic inside log_order; test_agent.csv with a pass rate

Modes (python task.py <mode>)
-----------------------------
faq      : store the FAQs in ChromaDB, search by meaning, then UPDATE a fact (no API key needed)
chunk    : chunking demo - the same policy text cut at different sizes (no API key needed)
memory   : two customers in two threads + a third thread that must not remember them
edge     : test it like a customer - own-words question, not in the FAQs, follow-up, thread switch
tests    : sign-off - run test_agent.csv and print the pass rate (goal: at least 80%)
chat     : interactive chat (type bye to stop; type 'thread <name>' to switch customer)
all      : faq, chunk, memory, edge, tests   (default)

Setup
-----
    pip install langchain langchain-groq gspread chromadb
    export GROQ_API_KEY="your-key"       # Windows PowerShell: $env:GROQ_API_KEY="your-key"

  The first ChromaDB search downloads a small embedding model (~80 MB, no API key).
  Orders sheet backend (same as Day 4): Colab sign-in -> service account
  (GOOGLE_SERVICE_ACCOUNT_FILE) -> local orders.csv fallback (force: ORDERS_BACKEND=local).
"""

import argparse
import csv
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

import chromadb
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
HERE = Path(__file__).parent
LOCAL_ORDERS = HERE / "orders.csv"
AGENT_TESTS = HERE / "test_agent.csv"
SHEET_NAME = "Sweet Crumbs Orders"
HEADER = ["name", "item", "qty", "date", "eggless", "total"]
PAUSE = float(os.environ.get("CALL_PAUSE", "1"))  # seconds between questions (rate limits)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def ensure_api_key() -> None:
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        from google.colab import userdata

        os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
    except Exception:
        raise SystemExit("GROQ_API_KEY not found. Set it as an environment variable.")


def make_llm() -> ChatGroq:
    return ChatGroq(model=MODEL, temperature=0.2, max_retries=3)  # retries 429 automatically


# --------------------------------------------------------------------------- #
# Day 4 pieces: data, Order model, orders sheet, tools
# --------------------------------------------------------------------------- #
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}
HOURS = {
    "monday": "9 am to 8 pm", "tuesday": "9 am to 8 pm", "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm", "friday": "9 am to 8 pm", "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm",
}


class Order(BaseModel):
    """A cake order taken from a customer's chat message."""

    name: str = Field(description="Customer's first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item")
    qty: int = Field(description="How many", ge=1)
    date: str = Field(description="Pickup date as the customer said it, e.g. '24 Dec'")
    eggless: bool = Field(description="True if the customer asked for eggless")


class LocalSheet:
    """Orders in a CSV file. Same interface as the Google backend."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write([HEADER])

    def _write(self, rows: list, mode: str = "w") -> None:
        with open(self.path, mode, newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)

    def append_row(self, row: list) -> None:
        self._write([row], mode="a")

    def row_count(self) -> int:
        with open(self.path, newline="", encoding="utf-8") as file:
            return max(sum(1 for _ in csv.reader(file)) - 1, 0)

    def describe(self) -> str:
        return f"local file {self.path.name}"


class GoogleSheet:
    def __init__(self, worksheet, url: str):
        self.ws, self.url = worksheet, url

    def append_row(self, row: list) -> None:
        self.ws.append_row(row)

    def row_count(self) -> int:
        return max(len(self.ws.get_all_values()) - 1, 0)

    def describe(self) -> str:
        return f"Google Sheet {self.url}"


def _google_client(gspread):
    try:  # inside Google Colab: sign in with a pop-up
        from google.auth import default
        from google.colab import auth

        auth.authenticate_user()
        creds, _ = default()
        return gspread.authorize(creds)
    except ImportError:
        pass
    key_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
    if not Path(key_file).exists():
        raise FileNotFoundError(f"service-account key '{key_file}' not found")
    return gspread.service_account(filename=key_file)


def connect_sheet():
    """Open (or create once) the orders sheet; fall back to a local CSV."""
    if os.environ.get("ORDERS_BACKEND", "").lower() == "local":
        return LocalSheet(LOCAL_ORDERS)
    try:
        import gspread

        gc = _google_client(gspread)
        try:
            sh = gc.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            email = os.environ.get("SHEET_SHARE_EMAIL")
            if email:
                sh.share(email, perm_type="user", role="writer")
        ws = sh.sheet1
        if not ws.row_values(1):
            ws.append_row(HEADER)
        return GoogleSheet(ws, sh.url)
    except Exception as error:
        print(f"[sheet] Google Sheets unavailable ({error}); using {LOCAL_ORDERS.name} instead.")
        return LocalSheet(LOCAL_ORDERS)


@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants
    """
    price = MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"


@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. "sunday"."""
    hours = HOURS.get(day.lower().strip())
    return f"Open {hours} on {day}." if hours else f"'{day}' is not a day of the week."


def make_log_order(sheet):
    @tool
    def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
        """Save a confirmed cake order to the Orders sheet.
        Use only when the customer has given their name, item, quantity and pickup date."""
        try:  # a bad row never reaches the sheet
            order = Order(name=name, item=item.lower().strip(), qty=qty, date=date, eggless=eggless)
        except Exception as error:
            return f"Could not save the order: {error}. Ask the customer to correct it."
        total = MENU[order.item] * order.qty
        sheet.append_row(
            [order.name, order.item, order.qty, order.date, "yes" if order.eggless else "no", total]
        )
        return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."

    return log_order


# --------------------------------------------------------------------------- #
# Knowledge: FAQs -> ChromaDB (RAG)
# --------------------------------------------------------------------------- #
FAQS = {
    "delivery": "We deliver within 5 km of the bakery. Delivery is Rs 50, free for orders above Rs 1,000.",
    "payment": "We accept UPI and cash. Orders above Rs 1,000 need a 50% advance by UPI.",
    "cancellation": "Cancellation is free up to 12 hours before pickup. After that, the advance is not refunded.",
    "photo_cake": "Photo cakes and custom designs need 48 hours' notice.",
    "notice": "Regular orders need 24 hours' notice.",
    "eggless": "All our cakes have eggless versions at the same price.",
    "allergens": "Plum cake contains nuts and dried fruit. Cupcakes and chocolate cake are nut-free.",
    "storage": "Keep cakes in the fridge and eat them within 3 days.",
    "message": "A message on the cake is free, up to 20 letters.",
    "candles": "Candles are free with every cake.",
    "discounts": "Bulk-order discounts need the owner's approval.",
    "pickup": "Pickup is during opening hours: Monday to Saturday 9 am to 8 pm, Sunday 10 am to 6 pm.",
}


def build_faq_collection(client=None, embedding_function=None, name: str = "sweet_crumbs_faq"):
    """Store the FAQs, one fact per chunk. upsert = add new ids, update existing ones."""
    client = client or chromadb.Client()  # in-memory database
    kwargs = {"embedding_function": embedding_function} if embedding_function else {}
    collection = client.get_or_create_collection(name=name, **kwargs)
    collection.upsert(ids=list(FAQS), documents=list(FAQS.values()))
    return collection


def chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into pieces of about `size` characters that overlap by `overlap`.

    Used for long documents. The bakery's FAQs are already one idea each, so they need no cutting.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):  # the last piece reached the end: stop (no tiny duplicate)
            break
        start += size - overlap
    return pieces


def make_search_faq(collection, n_results: int = 3):
    """Retrieval as a tool: the agent decides when a question is about a policy."""

    @tool
    def search_faq(question: str) -> str:
        """Search Sweet Crumbs' policies and FAQs: delivery, payment, cancellation,
        photo cakes, allergens, storage, candles and messages on cakes.
        Use for any question about rules or policies, not for prices or opening hours."""
        if collection.count() == 0:
            return "No policies are stored."
        results = collection.query(
            query_texts=[question], n_results=min(n_results, collection.count())
        )
        documents = results["documents"][0] if results["documents"] else []
        return "\n".join(documents) if documents else "No matching policy found."

    return search_faq


# --------------------------------------------------------------------------- #
# The complete single agent: tools + knowledge + memory
# --------------------------------------------------------------------------- #
SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
RULES:  Use get_price for prices and is_open for timings; never guess them.
        For any policy question, use search_faq and answer ONLY from what it returns.
        If the answer isn't there, say "I'll check with the owner."
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""


def text_of(message) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


class SingleAgent:
    """create_agent(model, tools, system_prompt, checkpointer): one thread_id per customer."""

    def __init__(self, llm, sheet, faq_collection, tools: list | None = None, system: str = SYSTEM):
        if tools is None:
            tools = [get_price, is_open, make_log_order(sheet), make_search_faq(faq_collection)]
        self.agent = create_agent(
            model=llm, tools=tools, system_prompt=system, checkpointer=InMemorySaver()
        )

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def chat(self, thread_id: str, text: str) -> str:
        """Send ONLY the new message; the checkpointer adds that thread's earlier messages."""
        if not text.strip():
            return "Please type your question and I'll help."
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": text}]}, self._config(thread_id)
            )
            return text_of(result["messages"][-1])
        except Exception as error:  # you see the real problem, the customer sees a polite line
            print("Agent error:", error)
            return "Sorry, I'm a little busy. Please try again in a minute."

    def messages(self, thread_id: str) -> list:
        state = self.agent.get_state(self._config(thread_id))
        return list(state.values.get("messages", []))


def trace_stats(messages: list) -> dict:
    ai_messages = [m for m in messages if getattr(m, "type", "") == "ai"]
    tools_used = [c["name"] for m in ai_messages for c in (getattr(m, "tool_calls", None) or [])]
    return {"model_calls": len(ai_messages), "tool_calls": len(tools_used), "tools": tools_used}


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def demo_faq(embedding_function=None) -> None:
    header("RAG step 1: store facts, search by MEANING (no keywords needed)")
    faq = build_faq_collection(embedding_function=embedding_function)
    print(faq.count(), "facts stored")
    for question in (
        "Can you send the cake to my house?",
        "Is it okay if I change my mind tomorrow?",
        "Will the plum cake hurt someone with a nut allergy?",
    ):
        found = faq.query(query_texts=[question], n_results=2)["documents"][0]
        print(f"\nQ: {question}")
        for rank, text in enumerate(found, 1):
            print(f"  {rank}. {text}")

    header("Update the DOCUMENT, not the model: delivery Rs 50 -> Rs 60")
    faq.upsert(ids=["delivery"], documents=[FAQS["delivery"].replace("Rs 50", "Rs 60")])
    print("After upsert:", faq.query(query_texts=["delivery charge?"], n_results=1)["documents"][0][0])
    print("(Same id = same fact, updated. The next answer would use Rs 60.)")


def demo_chunk() -> None:
    header("Chunking: one long policy page, three chunk sizes")
    page = " ".join(FAQS.values())
    print(f"Policy page: {len(page)} characters")
    for size, overlap in ((100, 20), (300, 50), (900, 100)):
        pieces = chunk(page, size, overlap)
        print(f"\nsize={size}, overlap={overlap}: {len(pieces)} chunks. First two:")
        for piece in pieces[:2]:
            print("   |", piece)
    print("\nToo small -> sentences split and lose meaning; too big -> mixed topics. "
          "Here one FAQ = one chunk is the natural choice.")


def demo_memory(agent: SingleAgent) -> None:
    header("Conversation memory: one thread per customer")
    script = [
        ("ravi", "Hi, I'm Ravi. I want 2 plum cakes."),
        ("sravani", "Hello! Do you make cupcakes?"),
        ("ravi", "Make them eggless please."),            # remembers: the plum cakes
        ("ravi", "What's my name, and will I pay for delivery? My order is about Rs 1,300."),
        ("asha", "What's my name?"),                       # a new thread: must NOT know Ravi
    ]
    for thread_id, message in script:
        print(f"\n[{thread_id}] {message}\n  Bot: {agent.chat(thread_id, message)}")
        time.sleep(PAUSE)
    for thread_id in ("ravi", "sravani", "asha"):
        print(f"thread {thread_id!r}: {len(agent.messages(thread_id))} saved messages")


def demo_edge(agent: SingleAgent) -> None:
    header("Test it like a customer")
    steps = [
        ("Policy question in my own words", "e1", "Could you bring the cake to my place, and is there an extra charge?"),
        ("Something NOT in the FAQs", "e2", "Do you sell biryani?"),
        ("Follow-up (setup)", "e3", "I'm Divya. I'd like 2 chocolate cakes for Sunday."),
        ("Follow-up that depends on the last message", "e3", "Is that one nut-free?"),
        ("Same follow-up in a NEW thread", "e4", "Is that one nut-free?"),
    ]
    for label, thread_id, message in steps:
        print(f"\n--- {label} [thread {thread_id}] ---\nCustomer: {message}\nBot: {agent.chat(thread_id, message)}")
        time.sleep(PAUSE)


def run_agent_tests(llm, faq_collection, path: Path = AGENT_TESTS, verbose: bool = True) -> dict:
    """Fresh thread per question (independent tests) and a throw-away orders sheet."""
    scratch = Path(tempfile.gettempdir()) / "sweet_crumbs_day5_orders.csv"
    scratch.unlink(missing_ok=True)
    sheet = LocalSheet(scratch)
    agent = SingleAgent(llm, sheet, faq_collection)
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    passed = 0
    for row in rows:
        before = sheet.row_count()
        reply = agent.chat(f"test-{uuid.uuid4().hex[:8]}", row["question"])
        text_ok = row["must_include"].lower() in reply.lower() if row["must_include"] else True
        rows_ok = sheet.row_count() - before == int(row["rows_added"])
        ok = text_ok and rows_ok
        passed += ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'} | {row['question']}\n      reply: {reply[:90]!r}  text_ok={text_ok} rows_ok={rows_ok}")
        time.sleep(PAUSE)
    rate = passed / len(rows) if rows else 0.0
    if verbose:
        print(f"\n{passed}/{len(rows)} passed ({rate:.0%})  ->  "
              f"{'sign-off goal (80%) reached' if rate >= 0.8 else 'below the 80% goal: fix RULES / tool docstrings and re-run'}")
    return {"passed": passed, "total": len(rows), "rate": rate}


def demo_tests(llm, faq_collection) -> None:
    header("Sign-off: run every test (goal: at least 80%)")
    run_agent_tests(llm, faq_collection)


def run_chat(agent: SingleAgent) -> None:
    thread_id = "customer-1"
    print("Sweet Crumbs assistant. Type 'bye' to stop, or 'thread <name>' to switch customer.")
    while True:
        text = input(f"[{thread_id}] You: ").strip()
        if text.lower() in ("bye", "exit", "quit"):
            print("Bot: Thank you for visiting Sweet Crumbs!")
            break
        if text.lower().startswith("thread "):
            thread_id = text.split(maxsplit=1)[1]
            print(f"(switched to thread {thread_id!r})")
            continue
        print("Bot:", agent.chat(thread_id, text))


MODES = ["faq", "chunk", "memory", "edge", "tests", "chat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 5 - Sweet Crumbs single agent: memory + RAG")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", *MODES])
    args = parser.parse_args()
    modes = [m for m in MODES if m != "chat"] if args.mode == "all" else [args.mode]

    llm = agent = faq = None
    if any(m in modes for m in ("memory", "edge", "tests", "chat")):
        ensure_api_key()
        llm = make_llm()
        faq = build_faq_collection()
        if any(m in modes for m in ("memory", "edge", "chat")):
            sheet = connect_sheet()
            agent = SingleAgent(llm, sheet, faq)

    for mode in modes:
        if mode == "faq":
            demo_faq()
        elif mode == "chunk":
            demo_chunk()
        elif mode == "memory":
            demo_memory(agent)
        elif mode == "edge":
            demo_edge(agent)
        elif mode == "tests":
            demo_tests(llm, faq)
        elif mode == "chat":
            run_chat(agent)


if __name__ == "__main__":
    main()
