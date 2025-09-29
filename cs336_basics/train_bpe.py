import json
from cs336_basics.tokenizer import run_train_bpe_adapter


def train_bpe_tinystories():
    vocab, merges = run_train_bpe_adapter("data/TinyStoriesV2-GPT4-valid.txt", 10_000, ["<|endoftext|>"])
    with open("data/tinystories_vocab.txt", "w") as f:
        for token in vocab.values():
            f.write(f"{token.decode('utf-8', errors='ignore')}\n")

    with open("data/tinystories_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")


def train_bpe_owt():
    vocab, merges = run_train_bpe_adapter("data/owt_train.txt", 32_000, ["<|endoftext|>"])
    with open("data/owt_vocab.txt", "w") as f:
        for token in vocab.values():
            f.write(f"{token.decode('utf-8', errors='ignore')}\n")

    with open("data/owt_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")


if __name__ == "__main__":
    # train_bpe_tinystories()
    train_bpe_owt()
