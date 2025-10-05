from dataclasses import dataclass, field
from heapq import heappop, heappush
from io import BytesIO
import functools
import json
import multiprocessing
import heapq
import os
import regex as re
from collections import defaultdict, Counter
from typing import BinaryIO, Iterable, Iterator
import time
import tqdm
import base64


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

TEXT = """
low low low<ctrl99>low low
lower lower widest widest widest<|endoftext|>
newest newest newest newest newest newest
"""


@dataclass
class PretokenInfo:
    freq: int
    pretoken: list[int]


@dataclass
class PairHeapElem:
    freq: int
    pair_bytes: tuple[bytes, bytes]
    pair: tuple[int, int]

    def __lt__(self, other):
        return (self.freq, self.pair_bytes, self.pair) > (other.freq, other.pair_bytes, other.pair)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(
    bounds: tuple[int, int], input_path: str | os.PathLike, special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    """Pretokenizes a chunk of text and returns a mapping of pretoken counts"""

    start, end = bounds
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    sections = re.split("|".join(map(re.escape, special_tokens)), chunk)
    pretoken_counts = Counter()

    for section in sections:
        for pretoken in re.finditer(PAT, section):
            pretoken_counts[tuple(pretoken.group().encode())] += 1

    return pretoken_counts


def _pretokenize_file(
    input_path: str | os.PathLike, num_processes: int, split_token: str, special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    """Pretokenizes the file and returns a mapping of pretokens counts"""

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, split_token)

    chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        # Note: move file reading to each worker to avoid blocking main thread
        all_pretoken_counts = pool.imap_unordered(
            functools.partial(_pretokenize_chunk, input_path=input_path, special_tokens=special_tokens), chunks
        )
        return sum(all_pretoken_counts, Counter())


def _get_pair_stats(
    pretoken_info: dict[int, PretokenInfo],
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    """Returns the count of each pair and a list of pretoken IDs that the pair was found in."""

    pair_freq = Counter()
    pair_occ = defaultdict(list)

    for pretoken_id, pretoken_info in pretoken_info.items():
        pretoken = pretoken_info.pretoken
        freq = pretoken_info.freq

        for token1, token2 in zip(pretoken, pretoken[1:]):
            pair = (token1, token2)
            pair_freq[pair] += freq
            pair_occ[pair].append(pretoken_id)

    return pair_freq, pair_occ


def _update_freq_and_occ(
    pretoken_info: dict[int, PretokenInfo],
    pair_freq: Counter[tuple[int, int]],
    # TODO: we should keep track of occurences as a set of pretokens to avoid recomputation.
    pair_occ: dict[tuple[int, int], list[int]],
    pair_heap: list[PairHeapElem],
    vocab: dict[int, bytes],
    token1: int,
    token2: int,
    new_token: int,
):
    """Updates the pretoken mapping and pair stats for a new merge"""

    for pretoken_id in pair_occ[(token1, token2)]:
        freq = pretoken_info[pretoken_id].freq
        pretoken = pretoken_info[pretoken_id].pretoken
        merged_pretoken = []

        i = 0
        while i < len(pretoken):
            if pretoken[i : i + 2] == [token1, token2]:
                # Decrement counts for old token pairs that will no longer exist once we merge the tokens.
                # In particular, if we have (x, a, b, y) and we'd like to merge (a, b) -> c, then:
                # - decrement (x, a) and (b, y)
                # - increment (x, c) and (c, y)
                if i > 0:
                    # We use merged_pretoken[-1] to handle the case where we have consecutive pairs that need
                    # to be merged. For example, for (x, a, b, a, b, y):
                    # first pass: - dec (x, a), (b, a); inc (x, c), (c, a). `merged_pretoken` so far = [x, c]
                    # second pass: dec (c, a), (b, y); inc (c, c), (c, y)
                    # end result: dec (x, a), (b, a), (b, y); inc (x, c), (c, c), (c, y)
                    # This avoids double subtracting the occurence for (b, a) in the middle.
                    pair = (merged_pretoken[-1], pretoken[i])
                    pair_freq[pair] -= freq
                    pair_occ[pair].remove(pretoken_id)
                    heappush(pair_heap, PairHeapElem(pair_freq[pair], (vocab[pair[0]], vocab[pair[1]]), pair))

                    new_pair = (merged_pretoken[-1], new_token)
                    pair_freq[new_pair] += freq
                    pair_occ[new_pair].append(pretoken_id)
                    heappush(
                        pair_heap,
                        PairHeapElem(pair_freq[new_pair], (vocab[new_pair[0]], vocab[new_pair[1]]), new_pair),
                    )

                if i + 2 < len(pretoken):
                    pair = (pretoken[i + 1], pretoken[i + 2])
                    pair_freq[pair] -= freq
                    pair_occ[pair].remove(pretoken_id)
                    heappush(pair_heap, PairHeapElem(pair_freq[pair], (vocab[pair[0]], vocab[pair[1]]), pair))

                    new_pair = (new_token, pretoken[i + 2])
                    pair_freq[new_pair] += freq
                    pair_occ[new_pair].append(pretoken_id)
                    heappush(
                        pair_heap,
                        PairHeapElem(pair_freq[new_pair], (vocab[new_pair[0]], vocab[new_pair[1]]), new_pair),
                    )

                merged_pretoken.append(new_token)
                i += 2
            else:
                merged_pretoken.append(pretoken[i])
                i += 1

        # Update the actual pretoken in the mapping
        pretoken_info[pretoken_id].pretoken = merged_pretoken

    # Get rid of the old token pair since we've already merged them
    del pair_freq[(token1, token2)]
    del pair_occ[(token1, token2)]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 32,
    split_token=b"<|endoftext|>",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Mapping from token ID to bytes. Initialized with bytes and special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode()

    next_idx = 256 + len(special_tokens)

    # A list of BPE merges ordered by creation.
    merges: list[tuple[bytes, bytes]] = []

    print(f"pretokenizing {input_path}")
    start = time.perf_counter()
    pretoken_counter = _pretokenize_file(input_path, num_processes, split_token, special_tokens)
    print("pretokenization took", time.perf_counter() - start)

    pretoken_info = {}
    for i, pretoken in enumerate(pretoken_counter):
        pretoken_info[i] = PretokenInfo(pretoken_counter[pretoken], list(pretoken))

    start = time.perf_counter()
    pair_freq, pair_occ = _get_pair_stats(pretoken_info)

    pair_heap = [PairHeapElem(freq, (vocab[pair[0]], vocab[pair[1]]), pair) for pair, freq in pair_freq.items()]
    heapq.heapify(pair_heap)

    for _ in tqdm.tqdm(range(vocab_size - len(vocab))):
        # Ignore stale frequencies
        while pair_freq[pair_heap[0].pair] != pair_heap[0].freq:
            heappop(pair_heap)

        head = heappop(pair_heap)
        token1, token2 = head.pair
        vocab[next_idx] = vocab[token1] + vocab[token2]
        merges.append((vocab[token1], vocab[token2]))

        _update_freq_and_occ(pretoken_info, pair_freq, pair_occ, pair_heap, vocab, token1, token2, next_idx)
        next_idx += 1

    assert len(vocab) == vocab_size
    print("training duration", time.perf_counter() - start)
    return vocab, merges


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        # Sort special tokens by descending length so that a special token that is a substring of another special token
        # is matched after the parent string.
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.rank = {
            (self.inv_vocab[token1], self.inv_vocab[token2]): rank for rank, (token1, token2) in enumerate(self.merges)
        }

        self.pat = re.compile(PAT)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {token_id: base64.b64decode(token) for token, token_id in vocab.items()}

        with open(merges_filepath) as f:
            merges = []
            for line in f.readlines():
                x, y = line.split(" ", 1)
                merges.append((base64.b64decode(x), base64.b64decode(y)))

        return cls(vocab, merges, special_tokens)

    def save(self, vocab_out: str | os.PathLike, merges_out):
        with open(vocab_out, "w") as f:
            json.dump({base64.b64encode(token).decode(): token_id for token_id, token in self.vocab.items()}, f)

        with open(merges_out, "w") as f:
            for merge in self.merges:
                f.write(f"{base64.b64encode(merge[0]).decode()} {base64.b64encode(merge[0]).decode()}\n")

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        if len(pretoken) == 0:
            return []

        tokens: list[int] = [self.inv_vocab[bytes([token])] for token in pretoken.encode()]

        while True:
            min_rank = float("inf")
            min_pair = None

            for token1, token2 in zip(tokens, tokens[1:]):
                r = self.rank.get((token1, token2))
                if r is not None and r < min_rank:
                    min_rank = r
                    min_pair = [token1, token2]

            if min_pair is None:
                break

            merged_token = self.inv_vocab[self.vocab[min_pair[0]] + self.vocab[min_pair[1]]]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if tokens[i : i + 2] == min_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sections = re.split("(" + "|".join(map(re.escape, self.special_tokens)) + ")", text)
        else:
            sections = [text]

        tokens = []
        for section in tqdm.tqdm(sections):
            if self.special_tokens and section in self.special_tokens:
                tokens.append(self.inv_vocab[section.encode()])
                continue

            for pretoken in re.finditer(self.pat, section):
                tokens.extend(self._encode_pretoken(pretoken.group()))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        return b"".join([self.vocab[token] for token in tokens]).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            yield from self.encode(item)


if __name__ == "__main__":
    t = Tokenizer(
        {
            0: b" ",
            1: b"a",
            2: b"c",
            3: b"e",
            4: b"h",
            5: b"t",
            6: b"th",
            7: b" c",
            8: b" a",
            9: b"the",
            10: b" at",
            11: b"<ctrl99>",
            12: b"<ctrl99><ctrl99>",
        },
        [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")],
        ["<ctrl99>", "<ctrl99><ctrl99>"],
    )

    print(t.encode("the cat <ctrl99><ctrl99>ate "))
