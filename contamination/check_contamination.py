import argparse
import json
import multiprocessing
import os
from typing import List, Dict, Tuple
import re
from functools import partial
from tqdm import tqdm  # Import tqdm for progress bars
from itertools import chain

try:
    from datasets import load_dataset
except ImportError:
    print(
        "Please install the 'datasets' library from Hugging Face to use HF datasets: pip install datasets"
    )
    exit(1)


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0.0


def tokenize(text: str) -> set:
    # Simple tokenization: split on whitespace and punctuation
    text = re.sub(r"[^\w\s]", "", text.lower())
    return set(text.split())


def check_entry(
    reference_data: List[Dict[str, str]],
    line_file: Tuple[str, str],
    q_key: str,
    a_key: str,
    threshold: float = 0.8,
) -> List[Tuple[Dict, Dict, str, float]]:
    contaminated = []
    line, input_file = line_file
    try:
        input_entry = json.loads(line.strip())
        input_q = input_entry.get("question", "").strip()
        input_a = input_entry.get("answer", "").strip()
        input_q_tokens = tokenize(input_q)
        input_a_tokens = tokenize(input_a)

        for ref_entry in reference_data:
            ref_q = ref_entry[q_key].strip()
            ref_a = ref_entry[a_key].strip()
            ref_q_tokens = tokenize(ref_q)
            ref_a_tokens = tokenize(ref_a)

            # Exact match for question or answer
            if input_q == ref_q or input_a == ref_a:
                contaminated.append((ref_entry, input_entry, "exact_match", 1.0))
                continue

            # Jaccard for question
            q_sim = jaccard_similarity(ref_q_tokens, input_q_tokens)
            if q_sim >= threshold:
                contaminated.append((ref_entry, input_entry, "jaccard_question", q_sim))
                continue

            # Jaccard for answer
            a_sim = jaccard_similarity(ref_a_tokens, input_a_tokens)
            if a_sim >= threshold:
                contaminated.append((ref_entry, input_entry, "jaccard_answer", a_sim))
    except json.JSONDecodeError:
        print(f"Invalid JSON in {input_file}: {line}")
    return contaminated


def get_all_lines(input_files: List[str]):
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    yield (line, input_file)


def main():
    parser = argparse.ArgumentParser(
        description="Contamination check between a reference dataset (GSM8K, MATH, etc.) and input JSONL datasets using exact match and Jaccard similarity."
    )
    parser.add_argument(
        "--reference-type",
        type=str,
        default="jsonl",
        choices=["jsonl", "hf"],
        help='Type of reference dataset: "jsonl" for local JSONL file or "hf" for Hugging Face dataset',
    )
    parser.add_argument(
        "--reference-jsonl",
        type=str,
        help="Path to reference JSONL file (required if --reference-type=jsonl)",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        help="Hugging Face dataset name (required if --reference-type=hf)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Split to load from HF dataset (default: train)",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="Key for question/problem in reference dataset entries",
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default="answer",
        help="Key for answer/solution in reference dataset entries",
    )
    parser.add_argument(
        "--input-jsonls",
        type=str,
        required=True,
        help="Comma-separated list of input JSONL files to check",
    )
    parser.add_argument(
        "--num-processes", type=int, default=16, help="Number of processes to use"
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for contamination",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="contamination_results.json",
        help="Output file for contamination results",
    )

    args = parser.parse_args()

    # Load reference data
    reference_data = []
    q_key = args.question_key
    a_key = args.answer_key

    if args.reference_type == "jsonl":
        if not args.reference_jsonl:
            parser.error("--reference-jsonl is required when --reference-type=jsonl")

        # Count lines for progress bar on reference loading
        num_ref_lines = sum(
            1 for _ in open(args.reference_jsonl, "r", encoding="utf-8")
        )

        with open(args.reference_jsonl, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=num_ref_lines, desc="Loading reference data"):
                try:
                    entry = json.loads(line.strip())
                    if q_key in entry and a_key in entry:
                        reference_data.append(entry)
                except json.JSONDecodeError:
                    print(f"Invalid JSON in reference JSONL: {line}")
    elif args.reference_type == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset is required when --reference-type=hf")
        dataset = load_dataset(
            path=args.hf_dataset,
            name=None if args.hf_dataset == "svc-huggingface/minerva-math" else "main",
            split=args.hf_split,
        )
        # For HF datasets, use tqdm on the iteration
        for entry in tqdm(dataset, desc="Loading reference data from HF"):
            if q_key in entry and a_key in entry:
                reference_data.append({q_key: entry[q_key], a_key: entry[a_key]})
            else:
                print(f"Entry missing keys {q_key} or {a_key}: {entry}")

    print(f"Loaded {len(reference_data)} reference entries.")

    # Input files
    input_files = [f.strip() for f in args.input_jsonls.split(",") if f.strip()]
    print(f"Checking {len(input_files)} input files.")

    # Calculate total number of lines for overall progress
    total_lines = sum(
        sum(1 for _ in open(f, "r", encoding="utf-8")) for f in input_files
    )
    print(f"Total input entries to process: {total_lines}")

    # Set up multiprocessing Pool
    num_processes = args.num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        worker_func = partial(
            check_entry,
            reference_data,
            q_key=q_key,
            a_key=a_key,
            threshold=args.jaccard_threshold,
        )
        # Use imap_unordered for memory efficiency and wrap with tqdm for progress
        results_iter = pool.imap_unordered(worker_func, get_all_lines(input_files))
        all_contaminated = []
        for res in tqdm(
            results_iter, total=total_lines, desc="Processing input entries"
        ):
            all_contaminated.extend(res)

    # Save results
    output_data = []
    for ref, inp, typ, score in all_contaminated:
        output_data.append(
            {"reference_entry": ref, "input_entry": inp, "type": typ, "score": score}
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(
        f"Found {len(all_contaminated)} contaminated entries. Results saved to {args.output}"
    )


if __name__ == "__main__":
    main()
