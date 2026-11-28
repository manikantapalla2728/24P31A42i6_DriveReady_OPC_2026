# Day 4 – Tools, structured output and Google Sheets

**Program:** Drive Ready – AI Agents | One Person Company (OPC) Initiative
**Case study:** Sweet Crumbs Home Bakery

## Task

Give the assistant hands. It now calls Python functions on its own to look up prices and opening hours and to save orders into a Google Sheet, using a Pydantic model so a bad order never reaches the sheet.

## Files

| File | Purpose |
|------|---------|
| `task.py` | Tools, `Order` model, sheet backends, agent, and all Day 4 experiments |
| `test_agent.csv` | Agent test cases: `question, must_include, rows_added` |

## What `task.py` implements

**Three tools** (`@tool`, snake_case name, type hints, docstring that says *when* to use it)

| Tool | Job | On bad input |
|------|-----|--------------|
| `get_price(item, qty)` | Total price from the `MENU` dict | Returns "'donut' is not on the menu. We have: …" |
| `is_open(day)` | Opening hours for a day | Returns "'funday' is not a day of the week." |
| `log_order(name, item, qty, date, eggless)` | Validates with Pydantic, works out the total, appends one row | Returns "Could not save the order: …. Ask the customer to correct it." |

**`Order` (Pydantic)**: `item` is a `Literal` of the three menu items and `qty` must be ≥ 1, so a donut or 0 cupcakes is rejected twice: once by `with_structured_output`, and again inside `log_order` as a second line of defence.

**Orders sheet**: opens `Sweet Crumbs Orders` if it exists (creates it once, header only once). Backends tried in order: Google Colab sign-in → service-account key (`GOOGLE_SERVICE_ACCOUNT_FILE`) → local `orders.csv`. All three expose the same `append_row` / `row_count`.

**Agent**: `create_agent(model, tools, system_prompt)` wrapped in an `OrderAgent` class that keeps the whole run as memory, ignores empty messages, and catches errors so the customer sees a polite line while the developer sees the real error. The model uses `max_retries=3` for 429 rate limits.

## Modes

| Command | Exercise from the lesson |
|---------|--------------------------|
| `python task.py build` | Build: ask hours + price, then book 2 eggless chocolate cakes, check the sheet row, print the trace |
| `python task.py edge` | Test it like a customer: donut, order with no date, 0 cupcakes. Nothing may be added to the sheet |
| `python task.py extract` | `with_structured_output(Order)` on 5 messy messages (one is deliberately incomplete) |
| `python task.py trace` | Count tool calls and model calls for one question |
| `python task.py break` | Rename a tool to `tool1` with docstring "does stuff"; compare which tools get called |
| `python task.py tests` | Run `test_agent.csv` through the agent against a throw-away local sheet |
| `python task.py chat` | Interactive chat |
| `python task.py` | build, edge, extract, trace, break, tests |

## How to run

**Google Colab (as taught):** add `GROQ_API_KEY` in Colab Secrets, upload `task.py` and `test_agent.csv`, then:

```python
!pip install -q langchain langchain-groq gspread
%run task.py build
```

Use `%run`, not `!python`: the Google sign-in pop-up and Colab Secrets only work inside the notebook's own process. The sign-in appears the first time.

**Locally:**

```bash
pip install -r ../requirements.txt
export GROQ_API_KEY="your-key"                      # Windows PowerShell: $env:GROQ_API_KEY="your-key"
export GOOGLE_SERVICE_ACCOUNT_FILE="service_account.json"   # optional; otherwise orders.csv is used
python task.py
```

Never commit `service_account.json`, `.env` or `orders.csv` (all are in `.gitignore`).

## Test cases (`test_agent.csv`)

| Question | Must include | Rows added | Checks |
|----------|--------------|------------|--------|
| Are you open on Sunday? | `10` | 0 | `is_open` used, no guessing |
| How much are 6 cupcakes? | `480` | 0 | `get_price` and exact maths |
| How much is a donut? | `menu` | 0 | Polite "not on the menu", no invented price |
| I'm Kiran. Book 2 plum cakes. | `date` | 0 | Asks for the missing date instead of saving half an order |
| I'm Meena. Book 0 cupcakes for 24 Dec. | – | 0 | Pydantic rejects qty 0; nothing saved |
| I'm Ravi. Book 2 eggless chocolate cakes for 24 Dec. | `1300` | 1 | Complete order saved with the right total |

Each test checks both the **words** in the reply and the **side effect** (rows added), because an agent can say the right thing and still do the wrong thing.

## Key learnings

- The model never runs code: it sends a tool call, **your code runs it**, and the result goes back as a tool message.
- The tool's name and docstring are the manual the model reads; unclear labels cause wrong tool picks.
- Tools return helpful messages on bad input instead of crashing.
- Pydantic turns free text into a checked record (`Literal`, `ge=1`); never fully trust model output that is about to be saved.
- 429 means rate limit: own key per student, small tests, `max_retries`, and `try/except` around the agent.
- Reading the trace (every message in `result["messages"]`) is how you debug what an agent really did.
- Every tool round costs another model call, so more tools can mean more tokens.

## My observations (fill in after running)

| Exercise | What I saw |
|----------|-----------|
| Build: row appeared in the sheet? | |
| Donut / no date / 0 cupcakes: what did the agent do? | |
| `extract`: which of the 5 messages failed or looked wrong? | |
| `trace`: model calls = ___, tool calls = ___ | |
| `break`: did `tool1` still get picked for "3 cupcakes"? | |
| `tests`: pass rate | ___ / 6 |
