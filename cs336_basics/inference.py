import dataclasses
import torch
from cs336_basics.training import TransformerConfig
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import Transformer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import load_checkpoint

import sys


def predict(transformer_config: TransformerConfig, device: str):
    # transformer_config.context_length = 1024
    model = Transformer(**dataclasses.asdict(transformer_config)).to(device)
    model = torch.compile(model, backend="aot_eager" if device == "mps" else "inductor")
    opt = AdamW(model.parameters(), lr=1e-3)

    load_checkpoint(
        sys.argv[1],
        model,
        opt,
    )

    tokenizer = Tokenizer.from_files("data/tinystories_vocab.json", "data/tinystories_merges.txt", ["<|endoftext|>"])
    eos_id = tokenizer.inv_vocab[b"<|endoftext|>"]
    text = "There was a"

    tokens = tokenizer.encode(text)
    out = model.generate(tokens, top_p=0.9, eos_id=eos_id, max_tokens=512, device=device)
    print(tokenizer.decode(out))


def main():
    config = TransformerConfig(
        d_model=512,
        num_heads=16,
        d_ff=1344,
        vocab_size=10_000,
        context_length=256,
        num_layers=4,
        theta=10_000,
    )
    predict(config, device="mps")


if __name__ == "__main__":
    main()
