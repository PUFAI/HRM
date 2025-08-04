#!/usr/bin/env python
"""
Example:
    python train_tokenizer.py \
        --repo HuggingFaceFW/fineweb \
        --split train \
        --sample_limit 100000 \
        --vocab_size 32000 \
        --output_dir tokenizer_sp
"""
import argparse
import tempfile
from pathlib import Path

import datasets
import sentencepiece as spm
from tokenizers import Tokenizer  
from transformers import PreTrainedTokenizerFast

def stream_to_text(repo: str, split: str, limit: int, tmp_path: Path):
    ds = datasets.load_dataset(repo, split=split, streaming=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if idx >= limit:
                break
            text = sample.get("text") or sample.get("content") or ""
            text = " ".join(text.split())
            if text:
                f.write(text + "\n")


def train_sentencepiece(corpus_path: Path, output_dir: Path, vocab_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "tokenizer"
    cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        "--model_type=bpe "
        "--character_coverage=1.0 "
        "--byte_fallback=true "
        "--unk_id=0 --pad_id=1 --bos_id=-1 --eos_id=-1 "
        "--input_sentence_size=1000000 "
        "--shuffle_input_sentence=true"
    )
    spm.SentencePieceTrainer.Train(cmd)
    return model_prefix.with_suffix(".model")


def export_fast_tokenizer(sp_model_path: Path, output_dir: Path):
    tok = Tokenizer.from_file(str(sp_model_path))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",  # id 0
        pad_token="<pad>",  # id 1
    )
    hf_tok.save_pretrained(output_dir)

def main():
    p = argparse.ArgumentParser(description="Train SentencePiece + export fast tokenizer")
    p.add_argument("--repo", required=True, help="HF dataset repo, e.g. HuggingFaceFW/fineweb")
    p.add_argument("--split", default="train", help="split name (streaming mode)")
    p.add_argument("--sample_limit", type=int, default=500_000, help="max examples to draw")
    p.add_argument("--vocab_size", type=int, default=32_000)
    p.add_argument("--output_dir", default="tokenizer_sp")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    print(f"Streaming ≈{args.sample_limit} lines from {args.repo}:{args.split} …")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    stream_to_text(args.repo, args.split, args.sample_limit, tmp_path)

    print("Training SentencePiece model …")
    sp_model_path = train_sentencepiece(tmp_path, out_dir, args.vocab_size)

    print("Exporting Hugging Face fast tokenizer …")
    export_fast_tokenizer(sp_model_path, out_dir)

    print("All files saved to", out_dir.resolve())


if __name__ == "__main__":
    main()
