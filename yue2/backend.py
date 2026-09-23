"""YuE2 AR attention surface.

Importing this module does not import the official runtime. ``--dummy`` uses a
tiny stand-in with the same projection names. A live run loads
``m-a-p/YuE2-3B`` through the pinned YuE2 runtime.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from yue2.defaults import (
    MODEL_ID,
    PROJECTIONS,
    UPSTREAM_REVISION,
    VAE_ID,
)
from yue2.prompts import sound_only


def require_yue2():
    try:
        from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Install the isolated YuE2 runtime: "
            f"yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@{UPSTREAM_REVISION}"
        ) from exc
    return YuE2Config, YuE2ForCausalLM


def architecture_of(model) -> dict:
    config = model.config
    model_type = getattr(config, "model_type", None)
    if model_type not in (None, "yue2"):
        raise ValueError("Expected a YuE2 AR model")
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "intermediate_size",
    )
    return {key: int(getattr(config, key)) for key in keys}


def _layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("YuE2 model has no decoder layers")
    return layers


def attention_targets(model) -> dict[str, nn.Linear]:
    architecture_of(model)
    targets: dict[str, nn.Linear] = {}
    for index, layer in enumerate(_layers(model)):
        attn = getattr(layer, "self_attn", None)
        for name in PROJECTIONS:
            module = getattr(attn, name, None)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Unsupported YuE2 projection: layer {index} {name}")
            targets[f"model.layers.{index}.self_attn.{name}"] = module
    expected = architecture_of(model)["num_hidden_layers"] * len(PROJECTIONS)
    if len(targets) != expected:
        raise ValueError("YuE2 attention layout does not match its configuration")
    return targets


class _TinyConfig:
    model_type = "yue2"

    def __init__(self, **values):
        defaults = dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            intermediate_size=64,
            max_position_embeddings=256,
        )
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, int(value))


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _TinyAttention(hidden)


class _TinyInner(nn.Module):
    def __init__(self, config: _TinyConfig):
        super().__init__()
        hidden = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.layers = nn.ModuleList(_TinyLayer(hidden) for _ in range(config.num_hidden_layers))
        self.norm = nn.LayerNorm(hidden)


class TinyYuE2(nn.Module):
    """CPU stand-in. Same AR projection names as YuE2, no Hub weights."""

    def __init__(self, config: _TinyConfig | None = None):
        super().__init__()
        self.config = config or _TinyConfig()
        self.model = _TinyInner(self.config)

    def forward_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(token_ids)
        # Causal running mean so the last state depends on the whole prefix.
        # A position-wise stack would ignore the caption whenever the final
        # history token matches.
        steps = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype)
        hidden = torch.cumsum(hidden, dim=1) / steps.view(1, -1, 1)
        for layer in self.model.layers:
            attn = layer.self_attn
            mixed = (attn.q_proj(hidden) + attn.k_proj(hidden) + attn.v_proj(hidden)) / 3
            hidden = hidden + attn.o_proj(mixed)
        return self.model.norm(hidden)


def tiny_from_architecture(record: dict) -> TinyYuE2:
    return TinyYuE2(_TinyConfig(**record))


class YuE2Backend:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        vae_id: str = VAE_ID,
        device: str = "cpu",
        allow_hub: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        dummy: bool = False,
    ):
        self.model_id = model_id
        self.vae_id = vae_id
        self.dummy = bool(dummy)
        self.allow_hub = bool(allow_hub)
        self.revision = revision
        self.cache_dir = cache_dir
        if dummy:
            self.model = TinyYuE2().eval()
            self.identity = {"dummy": True, "model_id": model_id}
            self.model.to("cpu").requires_grad_(False)
            return
        if model_id != MODEL_ID:
            raise ValueError(f"This product trains {MODEL_ID}")
        config_cls, model_cls = require_yue2()
        from yue2.storage import model_identity, resolve_model
        from yue2.tokenization_yue2 import YuE2TextTokenizer

        path = resolve_model(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=not allow_hub,
        )
        self.identity = model_identity(path)
        self.tokenizer = YuE2TextTokenizer(path / "qwen.tiktoken")
        dtype = torch.float32 if str(device).startswith("cpu") else torch.bfloat16
        self.model = model_cls.from_pretrained(path, local_files_only=True, torch_dtype=dtype).eval()
        self.model.to(device).requires_grad_(False)
        # Silence unused-import warnings if a caller only needed the classes.
        del config_cls

    def prefix(self, style: str, lyrics: str, cot: str = "off") -> list[int]:
        sound_only(style)
        sound_only(lyrics)
        if self.dummy:
            import hashlib

            text = f"{style}\n[Lyrics]\n{lyrics}\n"
            raw = list(text.encode("utf-8"))
            # Long captions share a prefix. A hard cut would drop the edit.
            # A digest of the full text keeps that edit in the dummy ids.
            digest = list(hashlib.sha256(text.encode("utf-8")).digest())
            ids = (raw[:96] + digest) if len(raw) > 96 else raw + digest
            if len(ids) < 8:
                ids.extend([1] * (8 - len(ids)))
            return ids
        from yue2.protocol import SongRequest, token_prefixes

        return token_prefixes(SongRequest(style=style, lyrics=lyrics, cot=cot), self.tokenizer)

    def continuation(self, prefix: list[int], count: int, seed: int) -> list[int]:
        if count < 1:
            raise ValueError("continuation count must be positive")
        if self.dummy:
            generator = torch.Generator().manual_seed(int(seed))
            return torch.randint(0, 64, (count,), generator=generator).tolist()
        from yue2.protocol import Sampling
        from yue2.sampling import generate_tokens

        tokens, _, _ = generate_tokens(
            self.model,
            prefix,
            Sampling(min_tokens=count, max_tokens=count),
            seed,
            "semantic",
            use_cuda_graph=False,
            legacy_off=True,
        )
        return tokens

    def hidden(self, ids: list[int], *, checkpointing: bool = False) -> torch.Tensor:
        if not ids:
            raise ValueError("YuE2 sequence is empty")
        limit = int(getattr(self.model.config, "max_position_embeddings", 10**9))
        if len(ids) > limit:
            raise ValueError("YuE2 sequence exceeds model context")
        device = next(self.model.parameters()).device
        if self.dummy:
            tokens = torch.tensor([ids], device=device, dtype=torch.long)
            return self.model.forward_hidden(tokens)
        return _live_hidden(self.model, ids, device, checkpointing=checkpointing)


def _live_hidden(model, ids, device, *, checkpointing: bool) -> torch.Tensor:
    """Unpadded causal AR forward. Logits stay off the train graph."""
    from torch.utils.checkpoint import checkpoint

    inner = model.model
    tokens = torch.tensor([ids], device=device, dtype=torch.long)
    positions = torch.arange(len(ids), device=device)[None]
    if not checkpointing:
        return inner(input_ids=tokens, position_ids=positions, use_cache=False)[0]
    hidden = inner.embed_tokens(tokens)
    cos, sin = inner.rotary_emb(positions)
    for layer in inner.layers:
        hidden = checkpoint(layer, hidden, cos, sin, use_reentrant=False)
    return inner.norm(hidden)


def file_digest(path: str | Path) -> str:
    import hashlib

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
