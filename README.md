# LLM as a Judge — Evaluation Methods in NLP 2026

## Part 1: Dataset Construction

### Pipeline order

```
scripts/01_sample_articles.py    → data/raw/sampled_articles.jsonl
scripts/02_generate_bart.py      → data/outputs/bart_summaries.jsonl
scripts/03_generate_cohere.py    → data/outputs/cohere_summaries.jsonl
scripts/04_build_dataset.py      → data/final/dataset.jsonl
                                    data/final/system_map.json       
                                    annotation/annotation_sheet.csv
```

### Setup

```bash
pip install -r requirements.txt
```
