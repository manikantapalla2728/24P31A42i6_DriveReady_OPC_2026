"""
Day 2 - Prompts, memory and the Python agents need
Case study: Sweet Crumbs Home Bakery

What this file does
-------------------
* MENU as a Python dict, and a system prompt built from it (Role / Facts / Rules / Format)
* get_price() with type hints, a docstring and a @log_call decorator
* A Chatbot class with memory (history) and trimming (max_turns)
* try / except so the bot never crashes - it fails politely
* "Your 10%" experiments: prompt parts, few-shot classifier, JSON order, memory trimming

Run
---
    export GROQ_API_KEY="your-key"      # Windows PowerShell: $env:GROQ_API_KEY="your-key"
    python task.py                      # run all demos
    python task.py chat                 # interactive chatbot (type bye to stop)
    python task.py rules | fewshot | json | memory | price
"""

import argparse
import functools
import json
import os

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata

        return userdata.get("GROQ_API_KEY")
    except Exception:
        raise SystemExit("GROQ_API_KEY not found. Set it as an environment variable.")


def make_client() -> OpenAI:
    return OpenAI(api_key=get_api_key(), base_url=GROQ_BASE_URL)


# --------------------------------------------------------------------------- #
# Data: the menu as a dict
# --------------------------------------------------------------------------- #
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}


def build_system_prompt(include_rules: bool = True) -> str:
    """Role + Facts + Rules + Format. Rules can be removed to see what breaks."""
    parts = [
        "ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.",
        f"FACTS:  Menu with prices in rupees: {json.dumps(MENU)}\n"
        "        Orders need 24 hours' notice. All cakes have eggless versions.",
    ]
    if include_rules:
        parts.append(
            "RULES:  Never promise a delivery time. If it isn't in FACTS, say \"I'll check with the owner.\"\n"
            "        Remember the customer's name and order during the chat."
        )
    parts.append("FORMAT: Under 50 words, friendly, English or Telugu matching the customer.")
    return "\n".join(parts)


SYSTEM = build_system_prompt()


# --------------------------------------------------------------------------- #
# Functions, type hints, decorator
# --------------------------------------------------------------------------- #
def log_call(func):
    """Decorator: print every call and its result. `@log_call` means f = log_call(f)."""

    @functools.wraps(func)  # keep the original name and docstring
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [log] {func.__name__}{args} -> {result!r}")
        return result

    return wrapper


@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the Sweet Crumbs menu."""
    return MENU[item.lower().strip()] * qty


def safe_price(item: str, qty: int = 1) -> str:
    """try/except around get_price: an unknown item gives a polite message, not a crash."""
    try:
        return f"Rs {get_price(item, qty)}"
    except KeyError:
        return f"Sorry, '{item}' isn't on our menu."


# --------------------------------------------------------------------------- #
# Chatbot with memory + trimming + polite failures
# --------------------------------------------------------------------------- #
class Chatbot:
    """Keeps the conversation in `history` and resends it on every call (LLMs have no memory)."""

    def __init__(self, client: OpenAI, system: str = SYSTEM, max_turns: int = 8,
                 temperature: float = 0.3):
        self.client = client
        self.history = [{"role": "system", "content": system}]
        self.max_turns = max_turns  # how many of the latest messages to resend
        self.temperature = temperature

    def chat(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        # always keep the system message, then only the most recent messages
        recent = [self.history[0]] + self.history[1:][-self.max_turns:]
        try:
            response = self.client.chat.completions.create(
                model=MODEL, messages=recent, temperature=self.temperature
            )
            reply = (response.choices[0].message.content or "").strip()
        except RateLimitError:
            reply = "We're a little busy. Please try again in a minute."
        except APIConnectionError:
            reply = "I can't reach the server right now. Please try again shortly."
        except APIError:
            reply = "Something went wrong on our side. Please try again."
        self.history.append({"role": "assistant", "content": reply})
        return reply


def run_chat_loop(client: OpenAI) -> None:
    bot = Chatbot(client)
    print("Sweet Crumbs bot ready. Type 'bye' to stop.")
    print("Total for 6 cupcakes:", safe_price("cupcake", 6))
    while True:
        text = input("You: ")
        if text.lower().strip() in ("bye", "exit", "quit"):
            print("Bot: Thank you for visiting Sweet Crumbs!")
            break
        print("Bot:", bot.chat(text))


# --------------------------------------------------------------------------- #
# "Your 10%" experiments
# --------------------------------------------------------------------------- #
def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def demo_price() -> None:
    header("Functions + decorator + try/except (no API call)")
    print(safe_price("Plum Cake"))
    print(safe_price("cupcake", 6))
    print(safe_price("donut"))
    print("dict .get() never crashes:", MENU.get("donut", "donuts not available"))


def demo_rules(client: OpenAI) -> None:
    """Delete the RULES and ask for delivery in 1 hour; then put them back."""
    header("Prompt parts: with and without RULES")
    question = "Can you deliver in 1 hour?"
    for label, include_rules in (("WITHOUT rules", False), ("WITH rules", True)):
        bot = Chatbot(client, system=build_system_prompt(include_rules), temperature=0.2)
        print(f"\n--- {label} ---\n{bot.chat(question)}")


FEW_SHOT = [
    {"role": "system",
     "content": "Label the customer message with one word: order, question or complaint."},
    {"role": "user", "content": "need 2 plum cakes for friday"}, {"role": "assistant", "content": "order"},
    {"role": "user", "content": "do you have eggless?"}, {"role": "assistant", "content": "question"},
    {"role": "user", "content": "the cake was dry yesterday"}, {"role": "assistant", "content": "complaint"},
]


def classify(client: OpenAI, message: str) -> str:
    """Few-shot classifier: three worked examples fix the answer format."""
    response = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=FEW_SHOT + [{"role": "user", "content": message}],
    )
    return (response.choices[0].message.content or "").strip().lower()


def demo_fewshot(client: OpenAI) -> None:
    header("Few-shot classifier: order / question / complaint")
    tests = [
        "I want a chocolate cake for Sunday",
        "what is the price of a cupcake?",
        "my order arrived late and cold",
        "can you make a cake without eggs?",
        "book 12 cupcakes for tomorrow",
    ]
    for message in tests:
        print(f"{message!r:45} -> {classify(client, message)}")


def parse_order_json(text: str) -> dict | None:
    """json.loads inside try/except. Models sometimes wrap JSON in ``` fences, so strip them."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def demo_json(client: OpenAI) -> None:
    header("Asking for JSON your code can use")
    system = (
        "Extract the order from the customer message. Reply with ONLY a JSON object "
        'with keys: name (string), item (string), qty (integer), date (string), eggless (boolean). '
        "No explanation, no markdown."
    )
    message = "Hi, I'm Ravi. I need 2 plum cakes for Friday, eggless please."
    response = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
    )
    raw = (response.choices[0].message.content or "").strip()
    print("Raw reply  :", raw)
    order = parse_order_json(raw)
    if order is None:
        print("Could not parse the reply as JSON.")
        return
    print("Parsed dict:", order)
    item = str(order.get("item", ""))
    qty = int(order.get("qty", 1))
    print("Order total:", safe_price(item, qty))


def demo_memory(client: OpenAI) -> None:
    """max_turns=2: the bot forgets the name once it falls outside the window."""
    header("Memory trimming with max_turns = 2")
    bot = Chatbot(client, max_turns=2, temperature=0.2)
    for message in ("Hi, my name is Ravi", "Do you make eggless cakes?",
                    "How much is a plum cake?", "What is my name?"):
        print(f"You: {message}\nBot: {bot.chat(message)}\n")
    print("Why it forgot: only the latest 2 messages are sent, so the message containing "
          "the name was trimmed away before the last question.")


DEMOS_WITH_CLIENT = {
    "rules": demo_rules,
    "fewshot": demo_fewshot,
    "json": demo_json,
    "memory": demo_memory,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2 - Sweet Crumbs chatbot that remembers")
    parser.add_argument(
        "mode", nargs="?", default="all",
        choices=["all", "chat", "price", *DEMOS_WITH_CLIENT],
        help="what to run (default: all demos)",
    )
    args = parser.parse_args()

    if args.mode == "price":
        demo_price()
        return

    client = make_client()
    if args.mode == "chat":
        run_chat_loop(client)
    elif args.mode == "all":
        demo_price()
        for demo in DEMOS_WITH_CLIENT.values():
            demo(client)
    else:
        DEMOS_WITH_CLIENT[args.mode](client)


if __name__ == "__main__":
    main()
