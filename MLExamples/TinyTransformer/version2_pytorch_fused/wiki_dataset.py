#!/usr/bin/env python3
"""
Wikipedia dataset support for tiny_llama_v2.py and evaluate_model.py.

``WikipediaTextDataset`` streams real text from HuggingFace's ``wikimedia/wikipedia``,
tokenized with a pretrained HuggingFace tokenizer (default: gpt2). Also provides the
``--dataset`` CLI args (shared with the synthetic random dataset) and the
Wikipedia-thread exit-crash workaround.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def safe_exit(code: int = 0) -> None:
    """Force-exit, bypassing teardown that can crash with streaming-dataset threads."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class WikipediaTextDataset:
    """Real-text dataset streamed from wikimedia/wikipedia, tokenized with a pretrained
    HuggingFace tokenizer and chunked into fixed-length blocks."""

    def __init__(
        self,
        seq_length: int = 128,
        wiki_config: str = "20231101.en",
        num_docs: int = 3000,
        val_fraction: float = 0.05,
        cache_dir: str = "./wiki_cache",
        tokenizer_name: str = "gpt2",
    ):
        self.seq_length = seq_length
        self.wiki_config = wiki_config
        self.num_docs = num_docs
        self.val_fraction = val_fraction

        from tokenizers import Tokenizer

        print(f"Loading pretrained tokenizer '{tokenizer_name}' ...")
        try:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        except Exception:
            # No/unreliable internet (e.g. compute node): fall back to an already-cached copy.
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.get_vocab_size()

        cache_key = f"{wiki_config}_{num_docs}_{tokenizer_name.replace('/', '_')}"
        self.cache_path = Path(cache_dir) / cache_key
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.tokens_path = self.cache_path / "tokens.npz"

        articles = None
        if not self.tokens_path.exists():
            articles = self._fetch_articles()

        train_ids, val_ids = self._load_or_build_tokens(articles)
        self.train_blocks = self._chunk(train_ids)
        self.val_blocks = self._chunk(val_ids)

        print(
            f"WikipediaTextDataset ready: {len(self.train_blocks):,} train / "
            f"{len(self.val_blocks):,} val blocks of {seq_length} tokens "
            f"(tokenizer={tokenizer_name}, vocab={self.vocab_size})"
        )

    def _fetch_articles(self) -> List[str]:
        from datasets import load_dataset

        print(f"Streaming {self.num_docs} articles from wikimedia/wikipedia [{self.wiki_config}] ...")
        ds = load_dataset("wikimedia/wikipedia", self.wiki_config, split="train", streaming=True)
        texts = []
        for i, row in enumerate(ds):
            if i >= self.num_docs:
                break
            text = row.get("text", "")
            if text:
                texts.append(text)
        print(f"Streamed {len(texts)} articles ({sum(len(t) for t in texts):,} characters)")
        return texts

    def _load_or_build_tokens(self, articles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if self.tokens_path.exists():
            print(f"Loading cached tokenized corpus from {self.tokens_path}")
            npz = np.load(self.tokens_path)
            return npz["train_ids"], npz["val_ids"]

        split_idx = max(1, int(len(articles) * (1 - self.val_fraction)))
        train_articles = articles[:split_idx]
        val_articles = articles[split_idx:] or articles[-1:]

        print(f"Tokenizing {len(train_articles)} train / {len(val_articles)} val articles ...")
        train_ids = self._encode_articles(train_articles)
        val_ids = self._encode_articles(val_articles)

        np.savez_compressed(self.tokens_path, train_ids=train_ids, val_ids=val_ids)
        print(
            f"Tokenized corpus cached to {self.tokens_path} "
            f"({len(train_ids):,} train tokens / {len(val_ids):,} val tokens)"
        )
        return train_ids, val_ids

    def _encode_articles(self, articles: List[str]) -> np.ndarray:
        ids: List[int] = []
        for text in articles:
            ids.extend(self.tokenizer.encode(text).ids)
        return np.array(ids, dtype=np.int64)

    def _chunk(self, ids: np.ndarray) -> np.ndarray:
        block = self.seq_length + 1
        n_blocks = len(ids) // block
        if n_blocks == 0:
            raise ValueError(
                f"Not enough tokens ({len(ids)}) to form a single block of length {block}. "
                "Increase --wiki-num-docs or decrease --seq-len."
            )
        usable = ids[: n_blocks * block]
        return usable.reshape(n_blocks, block)

    def get_batch(self, batch_size: int, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data from the requested split ('train' or 'val')."""
        blocks = self.train_blocks if split == "train" else self.val_blocks
        replace = batch_size > len(blocks)
        indices = np.random.choice(len(blocks), batch_size, replace=replace)
        batch = blocks[indices]

        input_ids = torch.from_numpy(batch[:, :-1].copy())
        labels = torch.from_numpy(batch[:, 1:].copy())
        return input_ids, labels

    def get_val_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around get_batch(..., split='val')."""
        return self.get_batch(batch_size, split="val")

    def decode(self, ids) -> str:
        """Decode a list/tensor of token ids back into text."""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token ids."""
        return self.tokenizer.encode(text).ids


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset-selection CLI arguments shared across training scripts."""
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        type=str,
        choices=["random", "wikipedia"],
        default="random",
        help="Training data source: synthetic random tokens (default) or streamed Wikipedia text",
    )
    group.add_argument(
        "--wiki-config",
        type=str,
        default="20231101.en",
        help="HuggingFace wikimedia/wikipedia dataset config, e.g. 20231101.en or 20231101.simple",
    )
    group.add_argument(
        "--wiki-num-docs",
        type=int,
        default=3000,
        help="Number of Wikipedia articles to stream and tokenize",
    )
    group.add_argument(
        "--wiki-val-fraction",
        type=float,
        default=0.05,
        help="Fraction of streamed articles held out for validation",
    )
    group.add_argument(
        "--wiki-cache-dir",
        type=str,
        default="./wiki_cache",
        help="Directory to cache the tokenized corpus",
    )
    group.add_argument(
        "--wiki-tokenizer",
        type=str,
        default="gpt2",
        help="Pretrained HuggingFace tokenizer to use, e.g. gpt2, distilgpt2, bert-base-uncased",
    )
