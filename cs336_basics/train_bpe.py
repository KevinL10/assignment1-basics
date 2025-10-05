import json
import os
from cs336_basics.tokenizer import Tokenizer, train_bpe
import numpy as np
import numpy.typing as npt
from torch import special


def encode_and_save(tokenizer: Tokenizer, dataset_path: str | os.PathLike, out: str | os.PathLike):
    with open(dataset_path) as f:
        dataset = f.read()

    encoded = tokenizer.encode(dataset)
    print(f"Encoded {dataset_path} with {len(encoded)} tokens")

    np.save(out, np.array(encoded, dtype=np.uint16))


if __name__ == "__main__":
    dataset = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(dataset, 10_000, special_tokens)

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save("data/tinystories_vocab.json", "data/tinystories_merges.txt")

    # encode_and_save(tokenizer, dataset, "data/tinystories_encoded.npy")
