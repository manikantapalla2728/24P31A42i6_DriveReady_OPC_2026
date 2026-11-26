# Day 3 – Your first agent: what an agent is, and LangChain

**Program:** Drive Ready – AI Agents | One Person Company (OPC) Initiative
**Case study:** Sweet Crumbs Home Bakery

## Task

Rebuild the Sweet Crumbs assistant with LangChain, add tests that check it, and complete the "Your 30%" exercises: improve the prompt, test temperature stability, and decide where an agent is actually needed.

## Files

| File | Purpose |
|------|---------|
| `task.py` | LangChain assistant, memory, test runner, improvement and temperature experiments, agent preview |
| `test.csv` | Test cases: `question,must_include` |

## What `task.py` implements

- **Chain:** `prompt | llm | StrOutputParser()` with `ChatPromptTemplate`, `ChatGroq` (`openai/gpt-oss-120b`).
- **Memory:** `MessagesPlaceholder("history")`; the `Assistant` class appends each turn and sends it next time.
- **Tests:** `run_tests()` reads `test.csv`, asks each question with fresh history, and marks PASS if `must_include` appears in the answer. Prints the pass rate.
- **Improve:** `SYSTEM_V2` adds one rule (state the 24-hour notice clearly for same-day requests); both versions are tested and compared.
- **Switch:** runs the tests twice at `temperature=1.0` and lists which questions changed result.
- **Agent preview:** `create_agent(model, tools=[], system_prompt=SYSTEM)`. With no tools it behaves like the chain; tools arrive on Day 4.

| Command | What it runs |
|---------|--------------|
| `python task.py chat` | Memory demo ("I'm Ravi" → "What's my name?") |
| `python task.py tests` | Baseline test run |
| `python task.py improve` | Prompt v1 vs v2 pass rates |
| `python task.py temperature` | Two runs at temperature 1.0 |
| `python task.py agent` | LangChain agent preview |

## How to run

```bash
pip install -r ../requirements.txt
export GROQ_API_KEY="your-key"     # Windows PowerShell: $env:GROQ_API_KEY="your-key"
python task.py
```

## Test cases (`test.csv`)

| Question | Must include | Business rule checked |
|----------|--------------|-----------------------|
| How much is a plum cake? | `500` | Correct price |
| Do you make eggless cakes? | `eggless` | Eggless availability |
| Can I get a cake today evening? | `24 hours` | Notice period |
| Do you deliver to Kakinada? | `check with the owner` | No invented delivery promises |
| What is the price of 6 cupcakes? | `480` | Correct arithmetic (6 × 80) |

Good tests check the things that would cost the owner money if wrong.

## Agent or not? (three jobs at Sweet Crumbs)

| Job | Choice | Why |
|-----|--------|-----|
| Calculate the bill (price × quantity) | **Plain code** | Fixed rule with one right answer; no AI needed. |
| Answer "Do you make eggless cakes?" or write a Diwali offer post | **One LLM call** | Needs language ability but a single step and no outside actions. |
| Take an order: check the menu, check the date, log it, send a confirmation | **Agent** | Multiple steps with tools, where the next step depends on the previous result. |

Rule used: pick the simplest option that works.

## Key learnings

- A chatbot answers; an agent is given a goal and takes actions with tools until it is done.
- Four parts of an agent: brain (LLM), instructions (system prompt), tools (functions), memory (history).
- Agent loop: think → act → observe → repeat. The model asks for a tool, **your code runs it**, and the loop stops on a final answer or a round limit.
- LangChain gives one interface for many models plus building blocks: template, chain, parser.
- Tests turn "it seems fine" into a pass rate you can re-run after every change.

## My results (fill in after running)

| Run | Pass rate |
|-----|-----------|
| Baseline prompt (temp 0.3) | ___ / 5 |
| Improved prompt `SYSTEM_V2` | ___ / 5 |
| Temperature 1.0, run 1 | ___ / 5 |
| Temperature 1.0, run 2 | ___ / 5 |

**What I changed and what happened:** _(write one or two lines)_
