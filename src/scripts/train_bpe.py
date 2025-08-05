#!/usr/bin/env python
""" python train_bpe.py \
        --repo HuggingFaceFW/fineweb \
        --split train \
        --sample_limit 500000 \
        --vocab_size 32000 \
        --output_dir tokenizer_fast
"""
import argparse
import tempfile
from pathlib import Path

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def stream_to_tmp(repo: str, split: str, limit: int) -> Path:
    ds = datasets.load_dataset(repo, split=split, streaming=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    path = Path(tmp.name)
    with tmp as f:
        for idx, sample in enumerate(ds):
            if idx >= limit:
                break
            text = sample.get("text") or sample.get("content") or ""
            text = " ".join(text.split())
            if text:
                f.write(text + "\n")
    return path

def main():
    p = argparse.ArgumentParser(description="Train fast tokenizer (tokenizer.json) from HF stream")
    p.add_argument("--repo", required=True, help="HF dataset repo, e.g. HuggingFaceFW/fineweb")
    p.add_argument("--split", default="train", help="dataset split")
    p.add_argument("--sample_limit", type=int, default=500_000, help="number of rows to sample")
    p.add_argument("--vocab_size", type=int, default=32_000)
    p.add_argument("--output_dir", default="tokenizer_fast")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.sample_limit} lines from {args.repo}:{args.split} …")
    corpus_path = stream_to_tmp(args.repo, args.split, args.sample_limit)

    print("Training byte level BPE …")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<pad>"]
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    tokenizer.add_special_tokens(["<unk>", "<pad>"])
    tokenizer.post_processor = TemplateProcessing(
        single="$0",
        pair="$A $B",
        special_tokens=[("<unk>", tokenizer.token_to_id("<unk>"))],
    )

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>", pad_token="<pad>",
    )

    hf_tok.save_pretrained(out_dir)
    print("Saved tokenizer.json to", out_dir.resolve())


if __name__ == "__main__":
    main()
