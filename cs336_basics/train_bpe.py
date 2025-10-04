import json
import os
from cs336_basics.tokenizer import Tokenizer, train_bpe
import numpy as np
import numpy.typing as npt


def train_and_save(
    dataset: str | os.PathLike,
    vocab_out: str | os.PathLike,
    merges_out: str | os.PathLike,
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
) -> Tokenizer:
    vocab, merges = train_bpe(dataset, vocab_size, special_tokens)
    with open(vocab_out, "w") as f:
        json.dump({k: v.decode("utf-8", "ignore") for k, v in vocab.items()}, f)

    with open(merges_out, "w") as f:
        for merge in merges:
            f.write(f"{merge[0].decode("utf-8", "ignore")} {merge[1].decode("utf-8", "ignore")}\n")

    return Tokenizer(vocab, merges, special_tokens)


def encode_and_save(tokenizer: Tokenizer, dataset_path: str | os.PathLike, out: str | os.PathLike):
    with open(dataset_path) as f:
        dataset = f.read()

    encoded = tokenizer.encode(dataset)
    print(f"Encoded {dataset_path} with {len(encoded)} tokens")

    # print(dataset == tokenizer.decode(encoded))
    np.save(out, np.array(encoded, dtype=np.uint16))


if __name__ == "__main__":
    dataset = "data/TinyStoriesV2-GPT4-valid.txt"
    tokenizer = train_and_save(dataset, "data/tinystories_vocab.json", "data/tinystories_merges.txt")

    encode_and_save(tokenizer, dataset, "data/tinystories_encoded.npy")
