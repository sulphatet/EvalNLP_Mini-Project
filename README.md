# LLM as a Judge — Evaluation Methods in NLP 2026

## Part 1: Dataset Construction

### Pipeline order

```
scripts/01_sample_articles.py    → data/raw/sampled_articles.jsonl
scripts/02_generate_bart.py      → data/outputs/bart_summaries.jsonl
scripts/03_generate_cohere.py    → data/outputs/cohere_summaries.jsonl
scripts/04_build_dataset.py      → data/final/dataset.jsonl
                                    data/final/system_map.json       ← private
                                    annotation/annotation_sheet.csv
```

### Setup

```bash
pip install -r requirements.txt
```

### Step 1 — Sample articles

```bash
python scripts/01_sample_articles.py
```

Samples 50 articles from the CNN/DailyMail 3.0.0 **test** split.
Downloads the dataset on first run (~1 GB).

### Step 2 — Generate BART summaries

```bash
python scripts/02_generate_bart.py
```

Runs `facebook/bart-large-cnn` locally. On CPU this takes ~15–20 minutes.
On an M-series Mac (MPS) or CUDA GPU it is much faster.

> **Note**: BART was trained on CNN/DailyMail. Its outputs will be stylistically
> similar to the reference highlights. This in-domain advantage over GPT-3.5
> must be discussed in the paper.

### Step 3 — Generate Cohere summaries

```bash
export COHERE_API_KEY="..."
python scripts/03_generate_cohere.py
```

Requires a Cohere API key. Uses `command-a-03-2025` at `temperature=0.0`.
Script resumes from where it left off if interrupted.

### Step 4 — Build final dataset and annotation sheet

```bash
python scripts/04_build_dataset.py
```

Merges both sets of summaries, randomly assigns blind A/B labels, and outputs:
- `data/final/dataset.jsonl` — full dataset (includes system identities)
- `data/final/system_map.json` — which system is A vs B per example (**do not share with annotators**)
- `annotation/annotation_sheet.csv` — pre-filled sheet for annotators (blinded)

---

## Part 2: Human Evaluation

See [`annotation/guidelines.md`](annotation/guidelines.md) for annotator instructions.

Share with annotators:
- `annotation/annotation_sheet.csv`
- `annotation/guidelines.md`

Do **not** share:
- `data/final/system_map.json`
- `data/final/dataset.jsonl`

---

## Known limitations / things to discuss in the paper

1. **BART in-domain advantage**: `facebook/bart-large-cnn` was fine-tuned on CNN/DailyMail.
   Evaluating it on the same distribution is not a fair comparison with Cohere.
   Frame this as a design choice that creates an interesting asymmetry to study.

2. **Reference summary quality**: CNN/DailyMail highlights are journalist-written bullet
   points, not abstractive summaries. They are an imperfect gold reference.

3. **GPT-3.5 temperature=0**: Using deterministic outputs means each article gets exactly
   one GPT summary. This is correct for reproducibility but means you cannot measure
   GPT's output variance. Note this in the methodology.
