from cs336_basics.tokenizer import run_train_bpe_adapter


def train_bpe_tinystories():
    run_train_bpe_adapter("data/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"])


if __name__ == "__main__":
    train_bpe_tinystories()
