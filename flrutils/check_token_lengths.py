"""Check input token lengths for question generation without needing a GPU.

Usage:
    python evaluation/check_token_lengths.py \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --dataset_family mtob \
        --dataset flrutils/grammar_book_chunks_512_sliding.csv \
        --max_items 1000
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer
from core.model_configs import get_model_config
from evaluation.utils import get_rag_context


# Same prompt templates as sample_questions.py
def build_prompt(context: str, dataset_family: str) -> str:
    if dataset_family == "mtob":
        prefix = "Here is a chunk of text from a kalamang textbook:"
    elif dataset_family == "qasper":
        prefix = "Here is a scientific paper:"
    else:
        prefix = "Here is a chunk of text:"

    if dataset_family == "mtob":
        suffix = (
            "Please generate challenging five trivia questions based on this text. "
            "If there are translation examples available you can use them to ask the translated version in either direction. "
            "Focus on grammar and vocabulary more than details about the book. "
            "Do not make the questions multiple-choice. Do not assume that the person answering the questions has access to the book. "
            "The questions must be understandable without access to the text. Do not output anything except the questions "
            "and format your output as in the followimg example:\n"
            "<question>What is the capital of Japan?</question>\n"
            "<question>How many months are there in a year?</question>\n"
            "<question>What was the first name of Reagan?</question>\n"
            "<question>How many goals did Messi score during the calendar year 2012</question>\n"
            "<question>Where is the Santa Monica pier located?</question>"
        )
    elif dataset_family == "qasper":
        suffix = (
            "Please generate challenging five trivia questions based on this paper. "
            "Focus on understanding the research methodology, key findings, contributions, and conclusions. "
            "Do not make the questions multiple-choice. Do not assume that the person answering the questions has access to the paper. "
            "The questions must be understandable without additional context. Do not output anything except the questions "
            "and format your output as in the following example:\n"
            "<question>What is the capital of Japan?</question>\n"
            "<question>How many months are there in a year?</question>\n"
            "<question>What was the first name of Reagan?</question>\n"
            "<question>How many goals did Messi score during the calendar year 2012</question>\n"
            "<question>Where is the Santa Monica pier located?</question>"
        )
    else:
        suffix = (
            "Please generate challenging five trivia questions based on this text. "
            "Do not make the questions multiple-choice. Do not assume that the person answering the questions has access to the paragraph. "
            "Do not mention the book. The questions must be understandable without access to the text. "
            "Do not output anything except the questions and format your output as in the followimg example:\n"
            "<question>What is the capital of Japan?</question>\n"
            "<question>How many months are there in a year?</question>\n"
            "<question>What was the first name of Reagan?</question>\n"
            "<question>How many goals did Messi score during the calendar year 2012</question>\n"
            "<question>Where is the Santa Monica pier located?</question>"
        )

    return f"{prefix}\n{context}\n\n{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_family", default="mtob")
    parser.add_argument("--dataset", default="flrutils/grammar_book_chunks_512_sliding.csv")
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    cfg = get_model_config(args.base)
    print(f"Loading tokenizer for {cfg.vllm_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.vllm_model, trust_remote_code=True)

    # Load contexts the same way as sample_questions.py
    contexts: List[str] = []
    if args.dataset_family == "mtob":
        df = pd.read_csv(args.dataset)
        for i, row in df.iterrows():
            contexts.append(row["text"])
    elif args.dataset_family == "squadshifts":
        ds = load_dataset("squadshifts", args.dataset, trust_remote_code=True)["test"]
        for i, item in enumerate(ds):
            if args.max_items and i >= args.max_items:
                break
            contexts.extend(get_rag_context(item, dataset_family=args.dataset_family))
    elif args.dataset_family == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", args.dataset, trust_remote_code=True)["validation"]
        for i, item in enumerate(ds):
            if args.max_items and i >= args.max_items:
                break
            contexts.extend(get_rag_context(item, dataset_family=args.dataset_family))
    elif args.dataset_family == "qasper":
        ds = load_dataset("allenai/qasper", trust_remote_code=True)[args.dataset]
        for i, item in enumerate(ds):
            if args.max_items and i >= args.max_items:
                break
            contexts.extend(get_rag_context(item, dataset_family=args.dataset_family))
    else:
        raise NotImplementedError(f"Unknown dataset family '{args.dataset_family}'")

    print(f"\n{'='*60}", flush=True)
    print(f"Dataset family: {args.dataset_family}")
    print(f"Total contexts:  {len(contexts)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    ctx_token_lens = []
    prompt_token_lens = []

    for i, ctx in enumerate(contexts):
        user_content = build_prompt(ctx, args.dataset_family)
        # Replicate messages_to_prompt: system + user with chat template
        messages = [
            {"role": "system", "content": cfg.system_message},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        ctx_tokens = len(tokenizer.encode(ctx, add_special_tokens=False))
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        ctx_token_lens.append(ctx_tokens)
        prompt_token_lens.append(prompt_tokens)

        print(f"[{i:4d}] context={ctx_tokens:5d} tokens  prompt={prompt_tokens:5d} tokens  (context chars={len(ctx)})")

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY ({len(contexts)} contexts)", flush=True)
    print(f"  Context tokens  — min: {min(ctx_token_lens)}, max: {max(ctx_token_lens)}, "
          f"mean: {sum(ctx_token_lens)/len(ctx_token_lens):.0f}", flush=True)
    print(f"  Prompt tokens   — min: {min(prompt_token_lens)}, max: {max(prompt_token_lens)}, "
          f"mean: {sum(prompt_token_lens)/len(prompt_token_lens):.0f}", flush=True)
    print(f"{'='*60}", flush=True)

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ctx_token_lens, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Context token lengths")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x=sum(ctx_token_lens)/len(ctx_token_lens), color="red", linestyle="--", label="mean")
    axes[0].legend()

    axes[1].hist(prompt_token_lens, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Full prompt token lengths (with chat template)")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")
    axes[1].axvline(x=sum(prompt_token_lens)/len(prompt_token_lens), color="red", linestyle="--", label="mean")
    axes[1].legend()

    fig.suptitle(f"{args.dataset_family} — {len(contexts)} contexts", fontsize=14)
    plt.tight_layout()

    out_path = f"token_lengths_{args.dataset_family}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
