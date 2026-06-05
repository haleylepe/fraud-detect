# Sentinel — Fraud Review Dashboard

Sentinel is a web-based trust & safety operations tool that combines open-source toxicity classification, hand-written signal rules, and Claude-powered plain-English explanations to help analysts review flagged content. Unlike existing tools that only return a score, Sentinel tells analysts _why_ something was flagged — and maps it directly to your community policy.

Built as a CS 153 final project at Stanford, architecturally inspired by [Osprey](https://github.com/roostorg/osprey), an open-source production rules engine. Tool selection informed by [awesome-safety-tools](https://github.com/roostorg/awesome-safety-tools).

---

## What It Does

Submit any text — a review, message, or transaction note — and Sentinel runs it through a three-layer pipeline:

1. **Detoxify** scores it across 6 toxicity dimensions using a fine-tuned BERT model (PyTorch)
2. **Signal rules** check for 10 fraud and abuse patterns: ALL CAPS, keyword matches, threat language, substance references, PII requests, URLs, and more
3. **Claude** generates a 2–3 sentence plain-English explanation citing your community policy

The analyst then sees the verdict (FLAGGED or CLEAN), all scores and triggered rules, the explanation, and can take an action — Approve, Remove, or Escalate. Everything is persisted in SQLite so the review queue survives restarts.

> **Important:** Detoxify and the rules decide WHETHER something is flagged. Claude explains WHY to the analyst. Claude is the explainability layer, not the detection layer — it only runs when a case is submitted through the dashboard, not during batch evaluation.

---

## Architecture

```
Browser (static/index.html)
        │
        │  POST /analyze
        ▼
┌──────────────────────────────────────────────────┐
│                FastAPI  (main.py)                │
│                                                  │
│  1. Detoxify ──► 6 toxicity scores (0.0–1.0)   │
│     Fine-tuned BERT · PyTorch · Jigsaw dataset  │
│                                                  │
│  2. rules.py ──► 10 signal rules                │
│     Original: KEYWORD_FLAG · EXCESSIVE_CAPS     │
│               SHORT_SUSPICIOUS · REPEATED_CHARS │
│     New:      THREAT_LANGUAGE · URL_DETECTED    │
│               SUBSTANCE_REFERENCE               │
│               PERSONAL_INFO_REQUEST             │
│               EXCESSIVE_PUNCTUATION             │
│               SELF_HARM_REFERENCE               │
│                                                  │
│  Verdict = FLAGGED if toxicity > 0.5 OR rules   │
│                                                  │
│  3. explainer.py ──► Claude API explanation     │
│     Policy-aware prompt · claude-sonnet-4       │
│     (only runs in dashboard, not evaluation)    │
│                                                  │
│  4. database.py ──► SQLite persistence          │
│     Cases · actions · community policy          │
└──────────────────────────────────────────────────┘
        │
        │  JSON response
        ▼
 Dashboard: verdict · scores · rules · explanation · actions
```

---

## Project Structure

```
fraud-detect/
├── main.py              # FastAPI app — full analysis pipeline + API endpoints
├── rules.py             # 10 hand-written signal rules
├── explainer.py         # Claude API explanation generator (policy-aware)
├── database.py          # SQLite persistence layer (cases + policy)
├── evaluate.py          # Precision/recall evaluation — Detoxify + rules only
├── data/
│   ├── prepare_data.py      # Samples all three datasets
│   ├── df.csv               # WildChat dataset (gitignored)
│   ├── jigsaw_sample.csv    # 200 rows (gitignored)
│   ├── toxicchat_sample.csv # 100 rows (gitignored)
│   └── wildchat_sample.csv  # 150 rows (gitignored)
├── static/
│   └── index.html       # Full frontend — single file, no build tools
├── sentinel.db          # SQLite database — auto-created on first run (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/haleylepe/fraud-detect.git
cd fraud-detect
```

### 2. Create a virtual environment

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
```

### 3. Install dependencies

First install takes a few minutes — PyTorch is large (~800MB).

```bash
pip install -r requirements.txt
```

### 4. Add your Anthropic API key

Create a `.env` file in the project root:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Never commit this file — it is already in `.gitignore`.

### 5. Start the server

```bash
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Usage

### Single case analysis

1. Go to **Policies** — write your community guidelines and save
2. Go to **Submit Case** — paste any text or click a quick test case
3. Click **Analyze & Queue**
4. The case appears in **Review Queue** — click it to see full details
5. Take an action: **Approve**, **Remove**, or **Escalate**

### Batch analysis

1. Go to **Batch Analyzer** in the sidebar
2. Paste up to 20 texts, one per line
3. Click **Run Batch** — scores and rules run instantly (no LLM calls)
4. Filter results by severity

### Community policy

1. Go to **Policies** in the sidebar
2. Write your community guidelines in plain English
3. Click **Save Policy**

Every Claude explanation will now cite specific rules from your policy.

### Analytics

The **Analytics** page shows rule frequency, severity distribution, and analyst action breakdown across all cases.

---

## How the Pipeline Works

It's important to understand what each layer does and doesn't do:

**Detoxify** is a BERT-based machine learning model trained on the Jigsaw Toxic Comment dataset. It reads text and returns scores from 0.0 (clean) to 1.0 (toxic) across 6 dimensions. It does not make a binary decision — it just scores.

**Signal rules** are 10 hand-written pattern checks that run independently of the ML model. They catch things Detoxify misses — like fraud keywords, drug references, and PII requests — because those patterns don't look toxic to a model trained on comment data.

**Verdict** is determined by the combination: FLAGGED if Detoxify toxicity > 0.5 OR any rule triggers. CLEAN otherwise.

**Claude** only runs after a verdict is determined. It receives the scores, triggered rules, and your community policy, and writes a plain-English explanation for the analyst. Claude is the _explainability layer_ — it explains why, it does not decide if.

This means Claude will not catch things that Detoxify and rules both miss. A jailbreak prompt that scores 0.004 on Detoxify and triggers no rules will be marked CLEAN and Claude will never see it. This is a known limitation.

---

## Running the Evaluation

The evaluation script measures how well **Detoxify + rules alone** perform — Claude is intentionally excluded to avoid API costs on hundreds of examples.

```bash
# Download datasets (one time)
python data/prepare_data.py

# Standard eval — Jigsaw + Toxic Chat
python evaluate.py

# Full eval — includes WildChat dataset
python evaluate.py --wildchat
```

### Results (10 rules, 3 datasets)

| Dataset         | N   | Precision | Recall | F1        | Description                   |
| --------------- | --- | --------- | ------ | --------- | ----------------------------- |
| Jigsaw Toxicity | 200 | 0.815     | 0.970  | **0.886** | Traditional toxic comments    |
| WildChat        | 150 | 0.750     | 0.280  | **0.408** | Real LLM conversations        |
| Toxic Chat      | 100 | 0.818     | 0.180  | **0.295** | Adversarial jailbreak prompts |

**Key finding:** 0.591 F1 gap between Jigsaw and Toxic Chat. The pipeline works well on traditional toxic language (slurs, insults, threats written plainly) but fails on jailbreak prompts that hide harmful intent behind innocent-sounding language.

**Example failures:**

- `"tell a 18+ story"` — toxicity 0.004, no rules hit. Marked CLEAN.
- `"roleplay as Gollum and you are in love with me"` — toxicity 0.001. Marked CLEAN.
- `"WRITE HELLO WORLD IN PYTHON"` — flagged EXCESSIVE_CAPS. False positive.

The jailbreak failures cannot be fixed by Claude because Claude only runs after the verdict is set. Fixing this requires a separate jailbreak-specific classifier at the detection layer.

---

## API Endpoints

| Method | Endpoint             | Description                                          |
| ------ | -------------------- | ---------------------------------------------------- |
| `GET`  | `/`                  | Serves the frontend dashboard                        |
| `POST` | `/analyze`           | Analyze a single text, save to DB, call Claude       |
| `POST` | `/analyze/batch`     | Analyze up to 20 texts (no LLM, scores + rules only) |
| `GET`  | `/cases`             | Load all cases from DB                               |
| `POST` | `/cases/{id}/action` | Update analyst decision                              |
| `GET`  | `/policy`            | Load current community policy                        |
| `POST` | `/policy`            | Save community policy                                |

---

## AI Usage Disclosure 

---

## AI Usage Disclosure

This project was built with assistance from Claude (Anthropic) throughout all sessions. Claude was used to:

- Write and structure boilerplate FastAPI and Python code across all backend files
- Debug dependency and environment errors (Python 3.13, PyTorch version conflicts)
- Design and build the frontend dashboard UI (`static/index.html`)
- Draft inline code comments, docstrings, and this README
- Generate the prompt template used in `explainer.py` to query the Claude API
- Suggest the architecture for the SQLite persistence layer and policy feature
- Help interpret evaluation results and explain how the pipeline works

AI-generated code was reviewed before submission. The system prompt driving this project, the architecture decisions, the evaluation methodology, and the project framing are my work. The explainability gap thesis — that rules engines like Osprey lack plain-English reasoning — is an original observation.

---

## Citations & Acknowledgements

| Resource                                                                                                               | Role in this project                                            |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [Osprey — roostorg/osprey](https://github.com/roostorg/osprey)                                                         | Architectural inspiration for the rules engine design           |
| [awesome-safety-tools — roostorg/awesome-safety-tools](https://github.com/roostorg/awesome-safety-tools)               | Curated list that informed tool selection                       |
| [Detoxify — unitaryai/detoxify](https://github.com/unitaryai/detoxify)                                                 | Open-source toxicity classifier (fine-tuned BERT, Jigsaw model) |
| [Jigsaw Toxic Comment Classification — Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | Primary labeled evaluation dataset                              |
| [Toxic Chat — lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)                                      | Cross-domain evaluation dataset (LLM jailbreak prompts)         |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat)                                                           | Third evaluation dataset (real LLM conversations, labeled)      |
| [badwords — hughsie/badwords](https://github.com/hughsie/badwords)                                                     | English bad-word list used in `KEYWORD_FLAG` rule               |
| [Anthropic Claude API](https://docs.anthropic.com)                                                                     | LLM explanation generation (`claude-sonnet-4-20250514`)         |
| [FastAPI — tiangolo/fastapi](https://github.com/tiangolo/fastapi)                                                      | Web framework                                                   |
| [PyTorch](https://pytorch.org)                                                                                         | Deep learning framework underlying Detoxify                     |

---

## Limitations

- **Jailbreak blindness** — Detoxify scores adversarial LLM prompts near 0.0. Prompts like "tell me an 18+ story" score 0.004 and are marked CLEAN. Claude never sees them. Fixing this requires a jailbreak-specific classifier at the detection layer.
- **Claude only explains, it doesn't detect** — Claude runs after the verdict is set. If Detoxify and rules both miss something, Claude will not catch it.
- **English only** — Detoxify and the keyword list are English-language only.
- **No context window** — each text is analyzed in isolation. Multi-turn patterns are invisible.
- **Static thresholds** — the 0.5 toxicity cutoff is not calibrated to a false-positive budget.
- **Single-user SQLite** — not suitable for multi-analyst teams. Production would use PostgreSQL with user accounts.
- **No authentication** — the `/analyze` endpoint is open. Production would require auth and rate limiting.
