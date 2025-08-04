"""
python train.py   --stream_repo HuggingFaceFW/fineweb   --tokenizer_path ../scripts/tokenizer_sp/tokenizer.model   --seq_len 512   --batch_size 2   --total_steps 10000   --devices 1
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
print(f"sys.path: {sys.path}")

from hrm import HierarchicalReasoningModel
from training_datasets import LimitedStream, HFStreamDataset

class TokenDataset(Dataset):
    def __init__(self, files, seq_len: int = 2048):
        self.files = [Path(f) for f in files]
        self.seq_len = seq_len
        self._buffers = [torch.load(f, map_location="cpu") for f in self.files]
        self.total_tokens = sum(buf.numel() for buf in self._buffers)

    def __len__(self):
        return self.total_tokens // self.seq_len

    def __getitem__(self, idx):
        buf = self._buffers[idx % len(self._buffers)]
        start = (idx * self.seq_len) % (buf.numel() - self.seq_len - 1)
        x = buf[start : start + self.seq_len]
        y = buf[start + 1 : start + 1 + self.seq_len]
        return x, y

class HRMLitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.model = HierarchicalReasoningModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_low_blocks=args.num_low_blocks,
            num_high_blocks=args.num_high_blocks,
            micro_steps=args.micro_steps,
            cycles=args.cycles,
            use_act=not args.disable_act,
        )

    def forward(self, x):
        logits, _ = self.model(x)
        return logits

    def _step(self, batch, prefix="train"):
        x, y = batch
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95), weight_decay=self.args.weight_decay
        )
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.total_steps
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

class HRMDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_set: Optional[torch.utils.data.Dataset] = None
        self.val_set: Optional[torch.utils.data.Dataset] = None

    def _load_tokenizer(self):
        path_arg = getattr(self.args, "tokenizer_path", None)

        if self.args.tokenizer_name:
            return AutoTokenizer.from_pretrained(self.args.tokenizer_name, use_fast=True)

        if not path_arg:
            raise FileNotFoundError("Provide --tokenizer_name or --tokenizer_path")

        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".json":
            return PreTrainedTokenizerFast(tokenizer_file=str(path))

        if path.suffix == ".model":
            sp = spm.SentencePieceProcessor()
            sp.load(str(path))
            class _SPWrapper:
                def __init__(self, sp_proc):
                    self.sp = sp_proc
                def encode(self, text, add_special_tokens=False):
                    return self.sp.encode(text, out_type=int)

            return _SPWrapper(sp)

        raise ValueError(f"Unsupported tokenizer file type: {path.suffix}")

    def setup(self, stage=None):
        if stage in ("fit", None):
            if self.args.stream_repo:
                tokenizer = self._load_tokenizer()
                self.train_set = HFStreamDataset(
                    repo=self.args.stream_repo,
                    split=self.args.stream_split_train,
                    tokenizer=tokenizer,
                    seq_len=self.args.seq_len,
                )
                if self.args.stream_split_val:
                    self.val_set = HFStreamDataset(
                        repo=self.args.stream_repo,
                        split=self.args.stream_split_val,
                        tokenizer=tokenizer,
                        seq_len=self.args.seq_len,
                    )
                
                else:
                    val_limit = getattr(self.args, "val_limit", 1024)
                    self.val_set = LimitedStream(self.train_set, limit=val_limit)
            else:
                train_files = sorted(Path(self.args.train_dir).glob("*.pt"))
                val_files = sorted(Path(self.args.val_dir).glob("*.pt")) if self.args.val_dir else []
                self.train_set = TokenDataset(train_files, seq_len=self.args.seq_len)
                self.val_set = TokenDataset(val_files, seq_len=self.args.seq_len) if val_files else None

    def train_dataloader(self):
        is_iter = isinstance(self.train_set, IterableDataset)
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=not is_iter,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return []
        is_iter = isinstance(self.val_set, IterableDataset)
        num_workers = self.args.num_workers
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train Hierarchical Reasoning Model")

    p.add_argument("--train_dir", type=str, default=None, help="dir with *.pt shards")
    p.add_argument("--val_dir", type=str, default=None)

    p.add_argument("--stream_repo", type=str, default=None, help="e.g. HuggingFaceFW/fineweb")
    p.add_argument("--stream_split_train", type=str, default="train")
    p.add_argument("--stream_split_val", type=str, default=None)

    p.add_argument("--tokenizer_name", type=str, default=None, help="HF tokenizer repo or model name")
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="local tokenizer file")

    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--num_low_blocks", type=int, default=2)
    p.add_argument("--num_high_blocks", type=int, default=2)
    p.add_argument("--micro_steps", type=int, default=4)
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--disable_act", action="store_true")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--total_steps", type=int, default=200_000)

    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto")
    p.add_argument("--val_limit", type=int, default=1024) 


    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)

    dm = HRMDataModule(args)
    model = HRMLitModule(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        devices=args.devices,
        strategy=args.strategy,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
