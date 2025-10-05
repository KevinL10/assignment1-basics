import torch.nn as nn
import torch
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        embeddings = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(embeddings, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_denom = torch.mean(x.square(), dim=-1, keepdim=True) + self.eps
        result = x * torch.rsqrt(rms_denom) * self.weight

        return result.to(in_dtype)


def silu(in_features: torch.Tensor) -> torch.Tensor:
    """Returns tensor after applying SiLU to each element"""
    return in_features * torch.sigmoid(in_features)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract max of elements for numerical stability
    x = x - torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * (self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k

        inv_freqs = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(0, max_seq_len, device=device)
        freqs = positions.unsqueeze(-1) * inv_freqs  # (max_seq_len, 1) * (d_k // 2, ) => (max_seq_len, d_k // 2)

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # sin, cos = (seq_len, d_k // 2, 1)
        cos = self.cos[token_positions].unsqueeze(-1)
        sin = self.sin[token_positions].unsqueeze(-1)

        # x = (..., seq_len, d_k)
        # x2 = (..., seq_len, d_k // 2, 2)
        x2 = x.view(*x.shape[:-1], self.d_k // 2, 2)

        # x0, x1 = (..., seq_len, d_k // 2, 1)
        x0 = x2[..., 0:1]
        x1 = x2[..., 1:2]

        # y0, y1 = (..., seq_len, d_k // 2, 1)
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        # y = (..., seq_len, d_k)
        y = torch.cat([y0, y1], dim=-1).view_as(x)

        return y


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    # Q: (..., q, d_k)
    # K: (..., k, d_k)
    # V: (..., k, d_v)
    d_k = Q.shape[-1]
    # scores: (..., q, k)
    scores: torch.Tensor = einsum(Q, K, ("... q d_k, ... k d_k -> ... q k")) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == False, -float("inf"))
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, ("... q k, ... k d_v -> ... q d_v"))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module | None = None):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * num_heads * self.d_k)
        self.output_proj = Linear(num_heads * self.d_v, d_model)

        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len))) == 1

        # qkv: (..., seq_len, 3 * num_heads * d_k)
        q, k, v = rearrange(
            self.qkv_proj(x),
            "... seq_len (c num_heads d_k) -> c ... num_heads seq_len d_k",
            c=3,
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            token_positions = torch.arange(0, seq_len, 1)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        heads = scaled_dot_product_attention(q, k, v, mask)
        heads = rearrange(heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.output_proj(heads)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None
    ):
        super().__init__()

        if theta and max_seq_len:
            rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        else:
            rope = None

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, theta, context_length) for _ in range(num_layers)]
        )
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    def _generate_token(self, x: torch.Tensor, temperature: float = 1.0, top_p: float | None = 0.9) -> torch.Tensor:
        # unbatched-generation: x has shape (L, )
        logits: torch.Tensor = self(x)[-1]  # (V, )

        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        probs = softmax(logits / temperature, dim=-1)

        if top_p:
            probs, sorted_indices = probs.sort(dim=-1, descending=True)
            mask = probs.cumsum(dim=-1) <= top_p
            if not mask.any():
                mask[0] = True

            probs *= mask
            probs /= probs.sum(dim=-1, keepdim=True)

            next_id = torch.multinomial(probs, 1)
            return sorted_indices[next_id]

        return torch.multinomial(probs, 1)

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        temperature: float = 1.0,
        top_p: float | None = 0.9,
        max_tokens: int = 100,
        eos_id: int | None = None,
    ) -> list[int]:
        x = torch.tensor(tokens, dtype=torch.int64)
        for _ in range(max_tokens):
            new_token = self._generate_token(x, temperature, top_p)
            x = torch.cat([x, new_token], dim=-1)

            if eos_id is not None and new_token.item() == eos_id:
                break

        return x.tolist()
