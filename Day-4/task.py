"""
Day 4 - Tools, structured output and Google Sheets
Case study: Sweet Crumbs Home Bakery

What this file does
-------------------
* Three tools the agent can call:  get_price, is_open, log_order
* A Pydantic `Order` model that checks every field before anything is saved
* An orders sheet: Google Sheets (Colab sign-in or service-account key) with an
  automatic local CSV fallback, so the script also runs without Google access
* An agent (create_agent) with memory, polite error handling and a readable trace

Modes (python task.py <mode>)
-----------------------------
build    : the chat from the lesson (hours + price, then book an order -> sheet row)
edge     : test it like a customer - donut, no date, 0 cupcakes; nothing may reach the sheet
extract  : with_structured_output(Order) on 5 messy messages
trace    : count tool calls and model calls for one question
break    : rename a tool to `tool1` with docstring "does stuff" and watch the agent
tests    : run test_agent.csv through the agent (uses a throw-away local sheet)
chat     : interactive chat (type bye to stop)
all      : build, edge, extract, trace, break, tests  (default)

Setup
-----
    pip install langchain langchain-groq gspread
    export GROQ_API_KEY="your-key"       # Windows PowerShell: $env:GROQ_API_KEY="your-key"

  Sheet backend (first one that works is used):
    1. Google Colab        -> a pop-up signs you in with Google
    2. Service account     -> GOOGLE_SERVICE_ACCOUNT_FILE=path/to/key.json
                              (share the sheet with the service account's e-mail, or set
                              SHEET_SHARE_EMAIL=you@gmail.com to share a newly created one)
    3. Local fallback      -> orders.csv next to this file (force with ORDERS_BACKEND=local)
"""

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
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
    # max_retries: LangChain retries rate-limit (429) errors a few times automatically
    return ChatGroq(model=MODEL, temperature=0.2, max_retries=3)


# --------------------------------------------------------------------------- #
# Data and the Order model (Pydantic = a form that checks itself)
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
    qty: int = Field(description="How many", ge=1)  # ge=1 -> at least 1
    date: str = Field(description="Pickup date as the customer said it, e.g. '24 Dec'")
    eggless: bool = Field(description="True if the customer asked for eggless")


# --------------------------------------------------------------------------- #
# Orders sheet: Google Sheets, with a local CSV fallback (same two methods)
# --------------------------------------------------------------------------- #
class LocalSheet:
    """Writes orders to a CSV file. Same interface as the Google backend."""

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
            return max(sum(1 for _ in csv.reader(file)) - 1, 0)  # minus the header

    def describe(self) -> str:
        return f"local file {self.path.name}"


class GoogleSheet:
    """Wraps a gspread worksheet."""

    def __init__(self, worksheet, url: str):
        self.ws = worksheet
        self.url = url

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
        pass  # not in Colab -> try a service-account key file
    key_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
    if not Path(key_file).exists():
        raise FileNotFoundError(f"service-account key '{key_file}' not found")
    return gspread.service_account(filename=key_file)


def connect_sheet():
    """Open (or create once) the 'Sweet Crumbs Orders' sheet; fall back to a local CSV."""
    if os.environ.get("ORDERS_BACKEND", "").lower() == "local":
        return LocalSheet(LOCAL_ORDERS)
    try:
        import gspread

        gc = _google_client(gspread)
        try:
            sh = gc.open(SHEET_NAME)  # re-use it, do not create a new sheet every run
        except gspread.exceptions.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            email = os.environ.get("SHEET_SHARE_EMAIL")
            if email:
                sh.share(email, perm_type="user", role="writer")
        ws = sh.sheet1
        if not ws.row_values(1):  # header only once
            ws.append_row(HEADER)
        return GoogleSheet(ws, sh.url)
    except Exception as error:
        print(f"[sheet] Google Sheets unavailable ({error}); using {LOCAL_ORDERS.name} instead.")
        return LocalSheet(LOCAL_ORDERS)


# --------------------------------------------------------------------------- #
# Tools: snake_case name + type hints + docstring that says WHEN to use it
# Bad input returns a helpful message; it never crashes the agent.
# --------------------------------------------------------------------------- #
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
    """Build the log_order tool for a given sheet (so tests can use a throw-away one)."""

    @tool
    def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
        """Save a confirmed cake order to the Orders sheet.
        Use only when the customer has given their name, item, quantity and pickup date."""
        try:  # second line of defence: a bad row never reaches the sheet
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
# The agent
# --------------------------------------------------------------------------- #
SYSTEM = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Always use get_price for prices and is_open for timings; never guess them.
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""


def text_of(message) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


class OrderAgent:
    """create_agent(model, tools, system_prompt) + memory + a last line of defence."""

    def __init__(self, llm, sheet, tools: list | None = None, system: str = SYSTEM):
        tools = tools if tools is not None else [get_price, is_open, make_log_order(sheet)]
        self.agent = create_agent(model=llm, tools=tools, system_prompt=system)
        self.history: list = []  # the whole run is kept as memory (questions, tool calls, results)

    def chat(self, text: str) -> str:
        if not text.strip():
            return "Please type your question and I'll help."
        try:
            result = self.agent.invoke(
                {"messages": self.history + [{"role": "user", "content": text}]}
            )
            self.history = result["messages"]
            return text_of(self.history[-1])
        except Exception as error:  # you see the real problem, the customer sees a polite line
            print("Agent error:", error)
            return "Sorry, I'm a little busy. Please try again in a minute."

    def trace(self) -> None:
        for message in self.history:
            message.pretty_print()  # question -> tool calls -> tool results -> answer


def trace_stats(messages: list) -> dict:
    """Count model calls (AI messages) and tool calls in an agent run."""
    ai_messages = [m for m in messages if getattr(m, "type", "") == "ai"]
    tools_used = [call["name"] for m in ai_messages for call in (getattr(m, "tool_calls", None) or [])]
    return {"model_calls": len(ai_messages), "tool_calls": len(tools_used), "tools": tools_used}


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def demo_build(llm, sheet) -> None:
    header(f"Build: chat -> tools -> sheet  ({sheet.describe()})")
    bot = OrderAgent(llm, sheet)
    before = sheet.row_count()
    print("Customer: Are you open on Sunday? How much are 2 chocolate cakes?")
    print("Bot:", bot.chat("Are you open on Sunday? How much are 2 chocolate cakes?"))
    time.sleep(PAUSE)
    print("\nCustomer: I'm Ravi. Please book 2 eggless chocolate cakes for 24 Dec.")
    print("Bot:", bot.chat("I'm Ravi. Please book 2 eggless chocolate cakes for 24 Dec."))
    print(f"\nRows added to the sheet: {sheet.row_count() - before} (expected 1)")
    print("\n--- Trace: what the agent actually did ---")
    bot.trace()


EDGE_CASES = [
    ("Donut (not on the menu)", "Can I order 3 donuts? I'm Asha, pickup Friday."),
    ("Order without a date", "Hi, I'm Kiran. Please book 2 plum cakes."),
    ("Zero cupcakes", "I'm Meena. Book 0 cupcakes for 24 Dec."),
]


def demo_edge(llm, sheet) -> None:
    header("Test it like a customer: nothing wrong may reach the sheet")
    for label, message in EDGE_CASES:
        bot = OrderAgent(llm, sheet)
        before = sheet.row_count()
        reply = bot.chat(message)
        added = sheet.row_count() - before
        print(f"\n[{'PASS' if added == 0 else 'FAIL'}] {label}\n  Customer: {message}\n  Bot: {reply}\n  Rows added: {added}")
        time.sleep(PAUSE)


MESSY_MESSAGES = [
    "hi ravi here, need 2 plum cakes for 24th dec, eggless pls",
    "Sravani: 6 cupcakes tomorrow evening",
    "arjun - one choco cake saturday, egg is fine",
    "bro need cake",  # missing almost everything on purpose
    "Meena wants twelve cupcakes by Friday, no eggs please",
]


def demo_extract(llm) -> None:
    header("Structured output: with_structured_output(Order) on messy messages")
    extractor = llm.with_structured_output(Order)
    for message in MESSY_MESSAGES:
        try:
            order = extractor.invoke(message)
            print(f"\n{message!r}\n  -> {order}")
        except Exception as error:  # Pydantic / the model refused to produce a valid Order
            print(f"\n{message!r}\n  -> REJECTED: {str(error).splitlines()[0][:110]}")
        time.sleep(PAUSE)


def demo_trace(llm, sheet) -> None:
    header("Read the trace: tool calls and model calls for ONE question")
    bot = OrderAgent(llm, sheet)
    question = "Are you open on Sunday? And how much for 2 chocolate cakes?"
    print("Question:", question)
    print("Answer  :", bot.chat(question))
    stats = trace_stats(bot.history)
    print(f"\nModel calls: {stats['model_calls']}   Tool calls: {stats['tool_calls']}   Tools: {stats['tools']}")
    print("(One round to ask for the tools, one more to write the answer from the results.)")


def demo_break(llm, sheet) -> None:
    header("Break it: rename a tool to `tool1` with the docstring 'does stuff'")

    @tool("tool1")
    def tool1(item: str, qty: int = 1) -> str:
        """does stuff"""
        price = MENU.get(item.lower().strip())
        return f"{qty} x {item} = Rs {price * qty}" if price else f"'{item}' is not on the menu."

    question = "How much are 3 cupcakes?"
    for label, tools in (
        ("CLEAR name + docstring (get_price)", [get_price, is_open]),
        ("VAGUE name + docstring (tool1)", [tool1, is_open]),
    ):
        bot = OrderAgent(llm, sheet, tools=tools)
        answer = bot.chat(question)
        print(f"\n--- {label} ---\nTools called: {trace_stats(bot.history)['tools']}\nAnswer: {answer}")
        time.sleep(PAUSE)
    print("\nThe model picks tools by reading names and docstrings, so vague labels risk wrong or missed picks.")


def run_agent_tests(llm, path: Path = AGENT_TESTS, verbose: bool = True) -> dict:
    """Each test: fresh agent + throw-away sheet. Checks the reply text AND rows added."""
    scratch = Path(tempfile.gettempdir()) / "sweet_crumbs_test_orders.csv"
    scratch.unlink(missing_ok=True)
    sheet = LocalSheet(scratch)
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    passed = 0
    for row in rows:
        bot = OrderAgent(llm, sheet)
        before = sheet.row_count()
        reply = bot.chat(row["question"])
        text_ok = row["must_include"].lower() in reply.lower() if row["must_include"] else True
        rows_ok = sheet.row_count() - before == int(row["rows_added"])
        ok = text_ok and rows_ok
        passed += ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'} | {row['question']}\n      reply: {reply[:90]!r}  text_ok={text_ok} rows_ok={rows_ok}")
        time.sleep(PAUSE)
    rate = passed / len(rows) if rows else 0.0
    if verbose:
        print(f"\n{passed}/{len(rows)} passed ({rate:.0%})")
    return {"passed": passed, "total": len(rows), "rate": rate}


def demo_tests(llm) -> None:
    header("Agent tests from test_agent.csv (throw-away local sheet)")
    run_agent_tests(llm)


def run_chat(llm, sheet) -> None:
    bot = OrderAgent(llm, sheet)
    print(f"Sweet Crumbs order assistant ({sheet.describe()}). Type 'bye' to stop.")
    while True:
        text = input("You: ")
        if text.lower().strip() in ("bye", "exit", "quit"):
            print("Bot: Thank you for visiting Sweet Crumbs!")
            break
        print("Bot:", bot.chat(text))


MODES = ["build", "edge", "extract", "trace", "break", "tests", "chat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4 - Sweet Crumbs agent with tools and a sheet")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", *MODES])
    args = parser.parse_args()

    ensure_api_key()
    llm = make_llm()
    needs_sheet = args.mode in ("all", "build", "edge", "trace", "break", "chat")
    sheet = connect_sheet() if needs_sheet else None

    runners = {
        "build": lambda: demo_build(llm, sheet),
        "edge": lambda: demo_edge(llm, sheet),
        "extract": lambda: demo_extract(llm),
        "trace": lambda: demo_trace(llm, sheet),
        "break": lambda: demo_break(llm, sheet),
        "tests": lambda: demo_tests(llm),
        "chat": lambda: run_chat(llm, sheet),
    }
    order = [m for m in MODES if m != "chat"] if args.mode == "all" else [args.mode]
    for mode in order:
        runners[mode]()


if __name__ == "__main__":
    main()
