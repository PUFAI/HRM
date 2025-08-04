#!/usr/bin/env python
"""
Example:
    python train_tokenizer.py \
        --repo HuggingFaceFW/fineweb \
        --split train[:1%] \
        --vocab_size 32000 \
        --output_dir tokenizer_sp
"""
import argparse
import tempfile
from pathlib import Path

import datasets
import sentencepiece as spm


def stream_to_text(repo: str, split: str, limit: int, tmp_path: Path):
    """Stream `limit` samples from `repo`/`split` and dump to `tmp_path`."""
    ds = datasets.load_dataset(repo, split=split, streaming=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if idx >= limit:
                break
            text = sample.get("text") or sample.get("content") or ""
            # normalise whitespace and write
            f.write(" ".join(text.split()) + "\n")


def train_sentencepiece(corpus_path: Path, output_dir: Path, vocab_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    spm_cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={output_dir / 'tokenizer'} "
        f"--vocab_size={vocab_size} "
        "--model_type=bpe "
        "--character_coverage=1.0 "
        "--byte_fallback=true "
        "--unk_id=0 --pad_id=1 --bos_id=-1 --eos_id=-1 "
        "--input_sentence_size=1000000 "
        "--shuffle_input_sentence=true"
    )
    spm.SentencePieceTrainer.Train(spm_cmd)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer from HF stream")
    parser.add_argument("--repo", type=str, required=True, help="HF dataset repo e.g. HuggingFaceFW/fineweb")
    parser.add_argument("--split", type=str, default="train[:1%]", help="split slice for training the tokenizer")
    parser.add_argument("--sample_limit", type=int, default=500_000, help="max number of samples to draw")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--output_dir", type=str, default="tokenizer_sp", help="where to write tokenizer.{model,vocab}")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Streaming {args.sample_limit} records from {args.repo}:{args.split} …")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    stream_to_text(args.repo, args.split, args.sample_limit, tmp_path)

    print("Training SentencePiece model …")
    train_sentencepiece(tmp_path, output_dir, args.vocab_size)

    print("Done! Files written to", output_dir.resolve())


if __name__ == "__main__":
    main()
