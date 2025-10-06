import os
from cs336_basics.tokenizer import Tokenizer, find_chunk_boundaries, train_bpe
import numpy as np
import multiprocessing
from functools import partial

TRAIN_DATASET_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_DATASET_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = "data/tinystories_vocab.json"
MERGES_PATH = "data/tinystories_merges.txt"

SPECIAL_TOKENS = ["<|endoftext|>"]


def encode_chunk(dataset_path: str | os.PathLike, chunk: tuple[int, int]) -> list[int]:
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    with open(dataset_path, "rb") as file:
        file.seek(chunk[0])
        chunk = file.read(chunk[1] - chunk[0]).decode("utf-8", errors="ignore")

    encoded = tokenizer.encode(chunk)
    return encoded


def encode_and_save(dataset_path: str | os.PathLike, out: str | os.PathLike, num_processes: int = 16):
    with open(dataset_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        encoded_chunks = pool.map(partial(encode_chunk, dataset_path), chunks)

    encoded = np.concatenate(encoded_chunks)
    print(
        "compression ratio",
    )
    np.save(out, np.array(encoded, dtype=np.uint16))


def train(dataset_path: str | os.PathLike = TRAIN_DATASET_PATH) -> Tokenizer:
    vocab, merges = train_bpe(dataset_path, 10_000, SPECIAL_TOKENS)
    return Tokenizer(vocab, merges, SPECIAL_TOKENS)


if __name__ == "__main__":
    # tokenizer = train(TRAIN_DATASET_PATH)
    # tokenizer.save(VOCAB_PATH, MERGES_PATH)

    encode_and_save(TRAIN_DATASET_PATH, "data/tinystories_train.npy")
    encode_and_save(VAL_DATASET_PATH, "data/tinystories_validation.npy")
