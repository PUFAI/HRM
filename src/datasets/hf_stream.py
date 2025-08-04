from pathlib import Path
from typing import Iterator, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


class HFStreamDataset(IterableDataset):
    """Stream text from a Hugging Face dataset and yield fixed length token windows.

    Args:
        repo: dataset identifier, e.g. "EleutherAI/fineweb".
        split: split name passed to `load_dataset`.
        tokenizer: a SentencePiece or `transformers.PreTrainedTokenizerFast` instance.
        seq_len: length of each sequence returned.
        streaming_kwargs: extra kwargs forwarded to `load_dataset`.

    Each __iter__ call creates a fresh iterator over the dataset stream, so you
    get different shards across dataloader workers.
    """

    def __init__(
        self,
        repo: str,
        split: str,
        tokenizer,
        seq_len: int = 2048,
        **streaming_kwargs,
    ):
        super().__init__()
        self.repo = repo
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.streaming_kwargs = streaming_kwargs

    def _tokenise_text(self, text: str):
        # tokenize into a torch LongTensor
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    def _chunk_generator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dset = load_dataset(
            self.repo,
            split=self.split,
            streaming=True,
            **self.streaming_kwargs,
        )
        buffer = []
        buf_len = 0
        for sample in dset:
            txt = sample.get("text") or sample.get("content") or ""
            token_ids = self._tokenise_text(txt)
            buffer.append(token_ids)
            buf_len += token_ids.size(0)
            # once buffer big enough to emit sequences
            while buf_len >= self.seq_len + 1:
                # concatenate lazily
                concat = torch.cat(buffer, dim=0)
                x = concat[: self.seq_len]
                y = concat[1 : self.seq_len + 1]
                yield x, y
                # trim buffer
                concat = concat[self.seq_len :]
                buffer = [concat]
                buf_len = concat.size(0)

    def __iter__(self):
        return self._chunk_generator()
