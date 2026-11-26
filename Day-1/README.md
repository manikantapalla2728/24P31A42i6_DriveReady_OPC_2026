# Day 1 – How LLMs work and your first API call

**Program:** Drive Ready – AI Agents | One Person Company (OPC) Initiative
**Case study:** Sweet Crumbs Home Bakery

## Task

Make a first API call to an LLM and give the bakery bot its facts and rules through a system message, then explore how an LLM behaves: temperature, hallucination, lack of memory and token cost.

## What `task.py` implements

| # | Experiment | Command | What it shows |
|---|-----------|---------|---------------|
| 1 | First answer | `python task.py first` | System prompt (Role / Facts / Rules / Format) + `temperature=0.2`; prints `response.usage` |
| 2 | Temperature | `python task.py temperature` | Same "tagline" request 3× at 0.2 and 3× at 1.2 |
| 3 | Hallucination | `python task.py hallucination` | "Do you deliver to Kakinada?" with and without the FACTS line |
| 4 | No memory | `python task.py memory` | "My name is Ravi" / "What is my name?" as two calls vs one `messages` list |
| 5 | Tokens | `python task.py tokens` | `prompt_tokens` for the same request in English vs Telugu |
| 6 | Bonus | `python task.py guard` | Prompt-injection classifier model (`llama-prompt-guard-2-22m`) |

## Implementation notes

- Uses the `openai` client pointed at Groq (`base_url="https://api.groq.com/openai/v1"`), because Groq speaks the OpenAI format. Only the key, URL and model name change.
- Model: `openai/gpt-oss-120b`.
- The API key is read from the `GROQ_API_KEY` environment variable (or Colab Secrets). It is **never** written in the code.
- The system prompt is split into **Role, Facts, Rules, Format**, so removing one part (experiment 3) shows exactly what that part prevents.

## How to run

```bash
pip install -r ../requirements.txt
export GROQ_API_KEY="your-key"     # Windows PowerShell: $env:GROQ_API_KEY="your-key"
python task.py                     # all experiments
```

## Key learnings

- An LLM predicts the next token in a loop; it does not look answers up.
- Low temperature gives focused, repeatable answers (facts, prices); high temperature gives varied, creative ones.
- Without FACTS the model fills gaps with plausible but invented answers; the fix is to put the real facts (and an "I'll check with the owner" rule) in the system message.
- The model has no memory between calls: the application must resend the conversation.
- Cost and limits are counted in tokens, and Telugu needs more tokens than English for the same meaning.

## My observations (fill in after running)

| Experiment | What I saw |
|-----------|------------|
| Temperature 0.2 vs 1.2 | |
| Kakinada without FACTS | |
| Kakinada with FACTS | |
| Telugu vs English `prompt_tokens` | English: ___  Telugu: ___ |
