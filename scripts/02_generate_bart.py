"""
Step 2: Generate summaries with facebook/bart-large-cnn (local inference).

BART is fine-tuned on CNN/DailyMail, so it has an in-domain advantage here.
This asymmetry is intentional — it creates an interesting contrast with GPT-3.5
and should be explicitly discussed in your paper (Part 6 / Part 7).

Input  : data/raw/sampled_articles.jsonl
Output : data/outputs/bart_summaries.jsonl
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_WORDS = 500   # BART's token limit is 1024; 500 words is a safe ceiling
MIN_SUMMARY_LEN = 56
MAX_SUMMARY_LEN = 142   # matches CNN/DM highlight length distribution


def truncate_to_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def resolve_device() -> str | int:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def main() -> None:
    articles_path = Path("data/raw/sampled_articles.jsonl")
    if not articles_path.exists():
        raise FileNotFoundError("Run 01_sample_articles.py first.")

    articles = [json.loads(l) for l in articles_path.read_text().splitlines()]

    device = resolve_device()
    device_label = {0: "CUDA", "mps": "MPS", -1: "CPU"}.get(device, str(device))
    print(f"Device : {device_label}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Loading model …")

    summariser = pipeline(
        "summarization",
        model=MODEL_NAME,
        device=device,
        truncation=True,
    )

    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    results = []

    for rec in tqdm(articles, desc="BART"):
        article = truncate_to_words(rec["article"], MAX_INPUT_WORDS)
        out = summariser(
            article,
            max_length=MAX_SUMMARY_LEN,
            min_length=MIN_SUMMARY_LEN,
            do_sample=False,
        )
        results.append(
            {
                "example_id": rec["example_id"],
                "bart_summary": out[0]["summary_text"].strip(),
            }
        )

    out_path = Path("data/outputs/bart_summaries.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} summaries → {out_path}")


if __name__ == "__main__":
    main()
