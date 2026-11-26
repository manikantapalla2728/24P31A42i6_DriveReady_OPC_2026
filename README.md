# Drive Ready – AI Agents | One Person Company (OPC) 2026

Daily practical tasks for the AI Agents training program (Department of Technical Training, Aditya University).
Case study throughout: **Sweet Crumbs Home Bakery**, an AI team built step by step.

## Progress

| Day | Topic | Folder |
|-----|-------|--------|
| 1 | How LLMs work and your first API call | [Day-1](Day-1) |
| 2 | Prompts, memory and the Python agents need | [Day-2](Day-2) |
| 3 | Your first agent, and LangChain | [Day-3](Day-3) |

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key"     # Windows PowerShell: $env:GROQ_API_KEY="your-key"
cd Day-1 && python task.py
```

Get a free key from [console.groq.com](https://console.groq.com). Never commit your key; it is read from the environment (or Colab Secrets).
