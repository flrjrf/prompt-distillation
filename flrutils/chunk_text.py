#!/usr/bin/env python3
"""
Text chunking script with two modes: greedy packing and sliding window.
Preserves inseparable parts (typically lines) while chunking.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package is required. Install with: pip install transformers")
    exit(1)


def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read text file with specified encoding."""
    return Path(file_path).read_text(encoding=encoding)


def prechunk_inseparable_parts_by_newline(text: str) -> List[str]:
    """Split input into inseparable parts (lines preserving newlines)."""
    return text.splitlines(keepends=True)


def encode_text(text: str, tokenizer, add_special_tokens: bool = False) -> List[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False)


def decode_token_ids(token_ids: List[int], tokenizer) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def greedy_pack_parts(
    parts: List[str],
    tokenizer,
    target_tokens: int,
    add_special_tokens: bool = False,
    show_progress: bool = True,
) -> List[Tuple[List[str], List[int]]]:
    """Greedily pack consecutive parts into groups bounded by token budget."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")

    min_chunk_tokens = max(1, int(target_tokens * 0.5))

    groups: List[Tuple[List[str], List[int]]] = []
    current_parts: List[str] = []
    current_text: str = ""
    current_token_ids: List[int] = []

    iterator = parts
    if show_progress:
        iterator = tqdm(parts, desc="Packing lines")

    for part in iterator:
        # Pre-tokenize the part
        part_ids = encode_text(part, tokenizer, add_special_tokens=add_special_tokens)
        part_len = len(part_ids)

        # If a single part is too large, flush current and emit it alone
        if part_len > target_tokens:
            if current_parts:
                # Prefer oversize over tiny chunks by attaching oversized part
                # when current chunk is still below minimum size.
                if len(current_token_ids) < min_chunk_tokens:
                    current_parts.append(part)
                    current_text = current_text + part
                    current_token_ids = encode_text(
                        current_text,
                        tokenizer,
                        add_special_tokens=add_special_tokens,
                    )
                    groups.append((current_parts, current_token_ids))
                else:
                    groups.append((current_parts, current_token_ids))
                    groups.append(([part], part_ids))
                current_parts, current_text, current_token_ids = [], "", []
            else:
                groups.append(([part], part_ids))
            continue

        # Test adding this part by tokenizing the joined group text
        if not current_parts:
            current_parts = [part]
            current_text = part
            current_token_ids = part_ids
        else:
            candidate_text = current_text + part
            candidate_ids = encode_text(candidate_text, tokenizer, add_special_tokens=add_special_tokens)

            if len(candidate_ids) <= target_tokens:
                current_parts.append(part)
                current_text = candidate_text
                current_token_ids = candidate_ids
            else:
                # Prefer oversize over tiny chunks when below minimum size.
                if len(current_token_ids) < min_chunk_tokens:
                    current_parts.append(part)
                    current_text = candidate_text
                    current_token_ids = candidate_ids
                else:
                    # Flush current group and start a new one
                    groups.append((current_parts, current_token_ids))
                    current_parts = [part]
                    current_text = part
                    current_token_ids = part_ids

    # Final flush
    if current_parts:
        # Merge tiny tail into previous chunk to avoid undersized leftovers.
        if groups and len(current_token_ids) < min_chunk_tokens:
            prev_parts, _ = groups.pop()
            merged_parts = prev_parts + current_parts
            merged_text = "".join(merged_parts)
            merged_ids = encode_text(
                merged_text,
                tokenizer,
                add_special_tokens=add_special_tokens,
            )
            groups.append((merged_parts, merged_ids))
        else:
            groups.append((current_parts, current_token_ids))

    return groups


def sliding_window_chunk(
    parts: List[str],
    tokenizer,
    target_tokens: int,
    overlap_ratio: float = 0.2,
    add_special_tokens: bool = False,
    show_progress: bool = True,
) -> List[Tuple[List[str], List[int]]]:
    """Chunk using greedy windows that restart with token overlap from prior end."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    min_chunk_tokens = max(1, int(target_tokens * 0.5))

    groups: List[Tuple[List[str], List[int]]] = []
    if not parts:
        return groups

    # Cache per-part token lengths for overlap backtracking.
    part_token_lens = [
        len(encode_text(part, tokenizer, add_special_tokens=add_special_tokens))
        for part in parts
    ]

    overlap_tokens = int(target_tokens * overlap_ratio)
    start_idx = 0

    progress = None
    if show_progress:
        progress = tqdm(total=len(parts), desc="Sliding windows")

    while start_idx < len(parts):
        window_parts: List[str] = []
        window_text = ""
        window_token_ids: List[int] = []

        i = start_idx
        while i < len(parts):
            part = parts[i]
            part_len = part_token_lens[i]

            # Oversized part is emitted as a single chunk.
            if part_len > target_tokens:
                if window_parts:
                    break
                window_parts = [part]
                window_token_ids = encode_text(
                    part,
                    tokenizer,
                    add_special_tokens=add_special_tokens,
                )
                i += 1
                break

            if not window_parts:
                window_parts = [part]
                window_text = part
                window_token_ids = encode_text(
                    part,
                    tokenizer,
                    add_special_tokens=add_special_tokens,
                )
                i += 1
                continue

            candidate_text = window_text + part
            candidate_ids = encode_text(
                candidate_text,
                tokenizer,
                add_special_tokens=add_special_tokens,
            )

            if len(candidate_ids) <= target_tokens:
                window_parts.append(part)
                window_text = candidate_text
                window_token_ids = candidate_ids
                i += 1
            else:
                break

        if not window_parts:
            # Defensive guard; should never trigger.
            raise RuntimeError("Failed to build sliding window chunk")

        # Avoid tiny chunks by allowing controlled oversize when below minimum.
        if len(window_token_ids) < min_chunk_tokens and i < len(parts):
            while i < len(parts) and len(window_token_ids) < min_chunk_tokens:
                candidate_text = window_text + parts[i]
                candidate_ids = encode_text(
                    candidate_text,
                    tokenizer,
                    add_special_tokens=add_special_tokens,
                )
                window_parts.append(parts[i])
                window_text = candidate_text
                window_token_ids = candidate_ids
                i += 1

        groups.append((window_parts, window_token_ids))
        end_idx = i

        # If this window consumed the end of input, we're done.
        if end_idx >= len(parts):
            if progress is not None:
                progress.update(len(parts) - progress.n)
                progress.close()
            break

        # Start next window from the beginning of the overlap context.
        if overlap_tokens <= 0:
            next_start = end_idx
        else:
            back_tokens = 0
            next_start = end_idx
            j = end_idx - 1
            while j >= start_idx:
                back_tokens += part_token_lens[j]
                next_start = j
                if back_tokens >= overlap_tokens:
                    break
                j -= 1

        # Ensure forward progress even with large overlap values.
        if next_start <= start_idx:
            next_start = start_idx + 1

        if progress is not None:
            progress.update(next_start - start_idx)
        start_idx = next_start

    if progress is not None and not progress.disable:
        progress.close()

    return groups


def save_results(
    groups: List[Tuple[List[str], List[int]]],
    tokenizer,
    output_json_path: str,
    output_csv_path: Optional[str] = None,
):
    """Save chunked results to JSON and optionally CSV."""
    # Decode groups back to text
    groups_text = [decode_token_ids(token_ids, tokenizer) for _, token_ids in groups]

    # Save JSON array
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(groups_text, f, ensure_ascii=False, indent=2)

    # Optional CSV for spreadsheets
    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "text"])
            for i, t in enumerate(groups_text):
                w.writerow([i, t])


def main():
    parser = argparse.ArgumentParser(
        description="Chunk text files with greedy packing or sliding window modes."
    )

    parser.add_argument(
        "input_file",
        help="Path to input text file"
    )

    parser.add_argument(
        "output_json",
        help="Path to output JSON file"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer model name (default: Qwen/Qwen3-4B-Instruct-2507)"
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=8192,
        help="Target token count per chunk (default: 8192)"
    )

    parser.add_argument(
        "--mode",
        choices=["greedy", "sliding"],
        default="greedy",
        help="Chunking mode: greedy (default) or sliding"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio for sliding window (0.0-0.9, default: 0.2)"
    )

    parser.add_argument(
        "--output-csv",
        help="Path to output CSV file (optional)"
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input file encoding (default: utf-8)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached tokenizer"
    )

    args = parser.parse_args()

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Read and preprocess input
    print("Reading input file...")
    text = read_text_file(args.input_file, encoding=args.encoding)
    parts = prechunk_inseparable_parts_by_newline(text)

    if len(parts) == 1 and parts[0] == "":
        raise ValueError("Input file is empty")

    print(f"Processing {len(parts)} parts with {args.mode} mode...")

    # Apply selected chunking mode
    if args.mode == "greedy":
        groups = greedy_pack_parts(
            parts,
            tokenizer,
            args.tokens,
            show_progress=not args.no_progress,
        )
    else:  # sliding
        groups = sliding_window_chunk(
            parts,
            tokenizer,
            args.tokens,
            overlap_ratio=args.overlap,
            show_progress=not args.no_progress,
        )

    # Save results
    print("Saving results...")
    save_results(
        groups,
        tokenizer,
        args.output_json,
        args.output_csv,
    )

    # Print summary
    print(f"\nSummary:")
    print(f"  Input file: {args.input_file}")
    print(f"  Mode: {args.mode}")
    print(f"  Target tokens: {args.tokens}")
    print(f"  Number of chunks: {len(groups)}")
    if groups:
        print(f"  Largest chunk: {max(len(g[1]) for g in groups)} tokens")
        print(f"  Smallest chunk: {min(len(g[1]) for g in groups)} tokens")


if __name__ == "__main__":
    main()
