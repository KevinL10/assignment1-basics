import os
import regex as re
from collections import defaultdict

from sympy.matrices.matrixbase import _jordan_form

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

TEXT = """
low low low<ctrl99>low low
lower lower widest widest widest<|endoftext|>
newest newest newest newest newest newest
"""


def train_bpe(
    text: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Mapping from token ID to bytes. Initialized with bytes and special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode()

    next_idx = 256 + len(special_tokens)

    # A list of BPE merges ordered by creation.
    merges: list[tuple[bytes, bytes]] = []

    # First, split by the special tokens. No merging should happen
    # across special tokens.
    sections = re.split("|".join(map(re.escape, special_tokens)), text)

    pretoken_counts = defaultdict(int)
    for section in sections:
        for pretoken in re.finditer(PAT, section):
            pretoken_counts[tuple(pretoken.group().encode())] += 1

    while len(vocab) < vocab_size:
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for pretoken, count in pretoken_counts.items():
            for token1, token2 in zip(pretoken, pretoken[1:]):
                pair_counts[(token1, token2)] += count

        # Take most common pair and break ties by choosing lexicographically greater pair
        token1, token2 = max(pair_counts, key=lambda k: (pair_counts[k], vocab[k[0]], vocab[k[1]]))

        vocab[next_idx] = vocab[token1] + vocab[token2]
        merges.append((vocab[token1], vocab[token2]))

        # Merge the existing pretokens
        new_pretoken_counts = defaultdict(int)
        for pretoken, count in pretoken_counts.items():
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if pretoken[i : i + 2] == (token1, token2):
                    new_pretoken.append(next_idx)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            new_pretoken_counts[tuple(new_pretoken)] += count

        pretoken_counts = new_pretoken_counts
        next_idx += 1

    return vocab, merges


def run_train_bpe_adapter(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path) as f:
        text = f.read()
    return train_bpe(text, vocab_size, special_tokens)


if __name__ == "__main__":
    vocab, merges = train_bpe(TEXT, 260, ["<ctrl99>", "<|endoftext|>"])
    print("merges:", merges)
