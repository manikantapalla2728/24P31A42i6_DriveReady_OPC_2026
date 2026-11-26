# Day 2 – Prompts, memory and the Python agents need

**Program:** Drive Ready – AI Agents | One Person Company (OPC) Initiative
**Case study:** Sweet Crumbs Home Bakery

## Task

Turn the Day 1 single answer into a real chatbot that follows the owner's rules, remembers the customer, turns orders into data and never crashes.

## What `task.py` implements

- **Menu as a dict** (`MENU`) and a system prompt built from it with the four parts: Role, Facts, Rules, Format.
- **`get_price(item, qty)`** – type hints, docstring, and a **`@log_call` decorator** that prints each call and result. `safe_price()` wraps it in `try/except KeyError` so an unknown item gets a polite message.
- **`Chatbot` class** – keeps `history`, resends it every call (memory), and trims to the latest `max_turns` messages (the system message is always kept).
- **Error handling** – `RateLimitError`, `APIConnectionError` and `APIError` each return a polite reply instead of crashing.
- **"Your 10%" experiments**

| Command | Experiment |
|---------|-----------|
| `python task.py rules` | Remove the RULES and ask "Can you deliver in 1 hour?", then put them back |
| `python task.py fewshot` | Few-shot classifier (order / question / complaint) tested on 5 new messages |
| `python task.py json` | Ask for an order as JSON (`name, item, qty, date, eggless`) and `json.loads` it inside `try/except` |
| `python task.py memory` | `max_turns=2`: tell it your name, chat for 3 turns, ask your name |
| `python task.py price` | Functions, decorator and `try/except` without any API call |
| `python task.py chat` | Interactive chatbot (type `bye` to stop) |

## How to run

```bash
pip install -r ../requirements.txt
export GROQ_API_KEY="your-key"     # Windows PowerShell: $env:GROQ_API_KEY="your-key"
python task.py                     # all demos
python task.py chat                # talk to the bot
```

Try it as a customer: say your name, ask for an eggless chocolate cake for tomorrow, ask the price for two, then ask "what did I order?". It should remember the name and order, quote Rs 1,300 and never promise a delivery time.

## Key learnings

- **Roles:** `system` = instructions, `user` = the person, `assistant` = the model's earlier replies.
- **Prompt parts:** a wrong answer usually means one of Role / Facts / Rules / Format is missing or unclear.
- **Few-shot:** worked examples make the answer format consistent so code can use it.
- **JSON:** ask for exact fields, then parse with `json.loads` inside `try/except` (models sometimes add code fences, so they are stripped first).
- **Memory is manual:** append every turn, resend it, and trim old turns to stay within the context window.
- **Python for agents:** dicts, functions with type hints and docstrings (the shape of every tool), decorators, and specific `except` clauses instead of a bare `except: pass`.

## My observations (fill in after running)

| Experiment | What I saw |
|-----------|------------|
| No RULES, "deliver in 1 hour?" | |
| With RULES, same question | |
| Few-shot: how many of 5 correct | |
| `max_turns=2`: why it forgot the name | |
