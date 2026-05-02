"""
Step 3: Generate summaries with Cohere command-a-03-2025 via the V2 API.

Requires: pip install cohere
Requires: COHERE_API_KEY environment variable set.

The prompt is deliberately minimal — no style instructions, no length targets.
Do NOT change the prompt between runs; record it verbatim in the paper appendix.

Input  : data/raw/sampled_articles.jsonl
Output : data/outputs/cohere_summaries.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a news summarisation assistant. "
    "Write accurate, concise summaries. Do not add information not present in the article."
)

USER_TEMPLATE = (
    "Summarise the following news article in 3 to 5 sentences. "
    "Include only information explicitly stated in the article.\n\n"
    "Article:\n{article}\n\nSummary:"
)

MODEL = "command-a-03-2025"
MAX_INPUT_WORDS = 800
RETRY_SLEEP = 5
MAX_RETRIES = 3


def truncate_to_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def call_with_retry(client, article: str) -> str:
    prompt = USER_TEMPLATE.format(article=truncate_to_words(article, MAX_INPUT_WORDS))
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"\n  API error (attempt {attempt}): {e}. Retrying in {RETRY_SLEEP}s …")
            time.sleep(RETRY_SLEEP)


def main() -> None:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY environment variable is not set.")
        print("  export COHERE_API_KEY='...'")
        sys.exit(1)

    try:
        import cohere
    except ImportError:
        print("ERROR: cohere package not installed.")
        print("  pip install cohere")
        sys.exit(1)

    articles_path = Path("data/raw/sampled_articles.jsonl")
    if not articles_path.exists():
        raise FileNotFoundError("Run 01_sample_articles.py first.")

    articles = [json.loads(l) for l in articles_path.read_text().splitlines()]

    out_path = Path("data/outputs/cohere_summaries.jsonl")
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-completed IDs
    completed: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            completed.add(json.loads(line)["example_id"])
        print(f"Resuming — {len(completed)} already done.")

    remaining = [r for r in articles if r["example_id"] not in completed]

    client = cohere.ClientV2(api_key=api_key)

    with open(out_path, "a") as f:
        for rec in tqdm(remaining, desc="Cohere"):
            summary = call_with_retry(client, rec["article"])
            row = {"example_id": rec["example_id"], "cohere_summary": summary}
            f.write(json.dumps(row) + "\n")

    total = len(completed) + len(remaining)
    print(f"\nSaved {total} summaries → {out_path}")
    print(f"Model: {MODEL}  temperature=0.0")
    print("Record this model version and temperature in your paper appendix.")


if __name__ == "__main__":
    main()
