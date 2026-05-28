# Fraud Review Dashboard

A web-based fraud and abuse detection dashboard that combines open-source toxicity classification (Detoxify), hand-written signal rules, and Claude-powered plain-English explanations. Built as a CS 153 final project, architecturally inspired by [Osprey](https://github.com/roostorg/osprey) — a production rules engine for trust & safety. Tool selection informed by [awesome-safety-tools](https://github.com/roostorg/awesome-safety-tools).

---

## Architecture

```
Browser (index.html)
        │
        │  POST /analyze  {"text": "..."}
        ▼
┌─────────────────────────────────────────────┐
│              FastAPI  (main.py)             │
│                                             │
│  1. Detoxify ──► toxicity scores (6 dims)  │
│  2. rules.py ──► triggered signal rules    │
│  3. explainer.py ──► Claude explanation    │
│                                             │
│  verdict = FLAGGED if score > 0.5 OR rules │
└─────────────────────────────────────────────┘
        │
        │  JSON response
        ▼
 Dashboard renders verdict, scores, rules, explanation
```

---

## Setup & Run

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd fraud-dashboard
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
```

### 3. Install dependencies

> First install takes 3–5 minutes — PyTorch (~800 MB) is the large one.

```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key

Create a `.env` file in the project root (never commit this):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Or export it in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Start the server

```bash
uvicorn main:app --reload
```

Open your browser to **http://localhost:8000**

---

## Running the Evaluation Script

> Requires Session 5 to be complete and data CSVs to exist in `data/`.

First, download and sample both datasets:

```bash
python data/prepare_data.py
```

Then run the evaluation:

```bash
python evaluate.py
```

This prints precision, recall, F1, and confusion matrices for both the Jigsaw Toxicity and Toxic Chat datasets, plus a cross-domain comparison table.

---

## Project Structure

```
fraud-dashboard/
├── main.py              # FastAPI app — all backend logic
├── rules.py             # Hand-written signal rules
├── explainer.py         # Anthropic Claude explanation generator
├── evaluate.py          # Precision/recall evaluation on labeled datasets
├── data/
│   ├── prepare_data.py      # Downloads + samples datasets from HuggingFace
│   ├── jigsaw_sample.csv    # 200 rows from Jigsaw Toxicity (primary eval)
│   └── toxicchat_sample.csv # 100 rows from Toxic Chat (cross-domain eval)
├── static/
│   └── index.html       # Entire frontend in one file
├── requirements.txt
└── README.md
```

---

## Credits

| Resource | Use in this project |
|---|---|
| [Osprey — roostorg/osprey](https://github.com/roostorg/osprey) | Architectural inspiration for the rules engine design |
| [awesome-safety-tools — roostorg/awesome-safety-tools](https://github.com/roostorg/awesome-safety-tools) | Curated list that informed tool selection |
| [Detoxify — unitaryai/detoxify](https://github.com/unitaryai/detoxify) | Open-source toxicity classifier (Jigsaw model) |
| [Jigsaw Toxicity — google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) | Primary labeled evaluation dataset |
| [Toxic Chat — lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | Cross-domain evaluation dataset (LLM interactions) |
| [badwords — hughsie/badwords](https://github.com/hughsie/badwords) | English bad-word list used in KEYWORD_FLAG rule |
| [Anthropic Claude API](https://docs.anthropic.com) | LLM explanation generation (claude-sonnet-4-20250514) |

---

## AI Usage Disclosure (CS 153 Requirement)

Claude (Anthropic) was used throughout this project to:
- Write and structure boilerplate FastAPI and Python code
- Debug errors when pasted tracebacks were provided
- Draft inline code comments and this README

All AI-generated code was reviewed, tested, and understood by the student before submission. The system prompt, evaluation methodology, architecture decisions, and project framing are the student's own work.

---

## Limitations

- **No context window**: each text is analyzed in isolation; multi-turn abuse patterns are invisible
- **English only**: Detoxify and the keyword list are English-language only; multilingual content will score near zero
- **Threshold is static**: the 0.5 Detoxify cutoff and rule triggers are not calibrated to a specific false-positive budget
- **LLM explanations can hallucinate**: Claude's explanation is informational, not a ground truth judgment
- **No authentication**: the `/analyze` endpoint is open; production would require auth + rate limiting
- **CPU inference latency**: Detoxify on CPU takes ~100–300ms per request; would need batching or GPU for high throughput
