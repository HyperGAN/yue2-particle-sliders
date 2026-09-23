"""Train a YuE2 AR slider on the shared winning formulation.

``python scripts/train_yue2.py --dummy`` runs on CPU and does not download Hub
weights. The game is ``winning_formulation()``: this file chooses YuE2 prompts,
token packing and which projections receive ``stamp.bridge()``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml

from yue2.backend import YuE2Backend
from yue2.defaults import CORE_PIN, EXAM_MODULE, MODEL_ID, VAE_ID
from yue2.prompts import load_prompts
from yue2.slider import ParticleSlider
from yue2.stamp import assert_product_game, locked_stamp

ROOT = Path(__file__).resolve().parents[1]
RUN_KEYS = {
    "name",
    "prompts_file",
    "model_id",
    "vae_id",
    "steps",
    "save_every",
    "seed",
    "device",
    "allow_hub",
    "revision",
    "cache_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train a routed-particle slider on m-a-p/YuE2-3B. "
            "The game is particle_sliders.winning_formulation(); "
            "--dummy does not download Hub weights."
        ),
    )
    parser.add_argument("--config_file", type=Path, default=ROOT / "configs" / "yue2" / "config-yue2-metal.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--prompts_file", type=Path, default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--vae_id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow_hub", action="store_true")
    parser.add_argument("--g_lr", type=float, default=None)
    parser.add_argument("--d_lr", type=float, default=None)
    parser.add_argument("--particle_lr", type=float, default=None)
    parser.add_argument("--adv_batch", type=int, default=None)
    parser.add_argument("--sample_seeds", type=int, default=None)
    parser.add_argument("--history_tokens", type=int, default=None)
    parser.add_argument("--seedbank_sources", type=int, default=None)
    parser.add_argument("--dummy", action="store_true", help="CPU stand-in, no Hub weights")
    parser.add_argument("--print_card", action="store_true", help="print the locked formulation card and exit")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    initial, _ = parser.parse_known_args(argv)
    if initial.config_file and Path(initial.config_file).exists() and not initial.print_card:
        loaded = yaml.safe_load(Path(initial.config_file).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            parser.error("config_file must be a mapping")
        _reject_forked_config(loaded)
        unknown = set(loaded) - RUN_KEYS - _surface_names()
        if unknown:
            parser.error(f"unsupported config keys: {sorted(unknown)}")
        parser.set_defaults(**loaded)
    args = parser.parse_args(argv)
    args.explicit_steps = _flag_set(argv, "--steps")
    args.name = args.name or "metal-yue2-particles"
    args.model_id = args.model_id or MODEL_ID
    args.vae_id = args.vae_id or VAE_ID
    args.prompts_file = Path(args.prompts_file or ROOT / "configs" / "yue2" / "prompts-yue2-metal-arm-b.yaml")
    args.save_dir = Path(args.save_dir) if args.save_dir else ROOT / "models" / args.name
    args.steps = 1200 if args.steps is None else int(args.steps)
    args.save_every = 100 if args.save_every is None else int(args.save_every)
    args.seed = 7 if args.seed is None else int(args.seed)
    args.device = "cuda:0" if args.device is None else str(args.device)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("name must be filename-safe")
    if args.model_id != MODEL_ID:
        parser.error(f"model_id must be {MODEL_ID}")
    if args.vae_id != VAE_ID:
        parser.error(f"vae_id must be {VAE_ID}")
    if min(args.steps, args.save_every) < 1 or not 0 <= args.seed < 2**63:
        parser.error("steps and save_every must be positive; seed must be in [0, 2**63)")
    return args


def _flag_set(argv: list[str] | None, flag: str) -> bool:
    if argv is None:
        return False
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def _surface_names() -> set[str]:
    from particle_sliders import winning_formulation

    return set(winning_formulation().model_surface_keys)


def _reject_forked_config(loaded: dict) -> None:
    from particle_sliders import winning_formulation

    stamp = winning_formulation()
    banned = set(stamp.formulation_keys) | set(stamp.provenance_keys) | set(stamp.architecture_keys)
    banned -= set(stamp.model_surface_keys)
    hit = sorted(banned & set(loaded))
    if hit:
        raise ValueError(
            f"config restates winning-formulation keys {hit}. "
            "Change the stamp in HyperGAN/particle-sliders."
        )


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = ROOT / path
    return candidate if candidate.exists() else path


def resolve_surface(args: argparse.Namespace, row_count: int) -> tuple[object, dict]:
    from particle_sliders import winning_formulation

    preview = winning_formulation()
    surface = {key: preview.as_dict()[key] for key in preview.model_surface_keys}
    for key in preview.model_surface_keys:
        value = getattr(args, key, None)
        if value is not None and key != "seedbank_sources":
            surface[key] = value
    if args.dummy:
        surface["adv_batch"] = min(int(surface["adv_batch"]), 2)
        surface["sample_seeds"] = 1
        surface["history_tokens"] = 4
        args.seedbank_sources = None
        if not args.explicit_steps:
            args.steps = 2
        args.device = "cpu"
        args.allow_hub = False
    for key in ("adv_batch", "sample_seeds", "history_tokens", "seedbank_sources"):
        if int(surface[key]) < 1:
            raise ValueError(f"{key} must be positive")
    bank = row_count * int(surface["sample_seeds"])
    if args.seedbank_sources is not None and int(args.seedbank_sources) != bank:
        raise ValueError(
            f"seedbank_sources {args.seedbank_sources} does not match "
            f"{row_count} rows x {surface['sample_seeds']} sample_seeds"
        )
    surface["seedbank_sources"] = bank
    if int(surface["adv_batch"]) > bank:
        raise ValueError("adv_batch cannot exceed the seed bank")
    return locked_stamp(surface)


def card_for(args: argparse.Namespace, stamp, locked: dict, *, rows: int) -> dict:
    return {
        "backend": "yue2",
        "product": "yue2-particle-sliders",
        "model_id": args.model_id,
        "vae_id": args.vae_id,
        "architecture_id": stamp.architecture_id,
        "formulation_id": stamp.formulation_id,
        "formulation_provisional": stamp.formulation_provisional,
        "recipe_name": locked["recipe_name"],
        "core_pin": CORE_PIN,
        "exam_module": EXAM_MODULE,
        "rows": rows,
        "steps": args.steps,
        "dummy": bool(args.dummy),
        "allow_hub": bool(args.allow_hub),
        "model_surface": {key: locked[key] for key in stamp.model_surface_keys},
        "non_goals": [
            "does not vendor RoutedMLP, GradRegularizer, locked_shared, or ParticleGAN",
            "does not fork the provisional exam; it stays in particle-sliders",
            "does not download Hub weights in --dummy",
        ],
    }


def prepare_bank(backend: YuE2Backend, rows: list[dict], *, sample_seeds: int, history_tokens: int, seed: int):
    limit = int(getattr(backend.model.config, "max_position_embeddings", 10**9))
    total = len(rows) * int(sample_seeds)
    generator = torch.Generator().manual_seed(int(seed) + 12345)
    drawn = torch.randint(0, 2**31 - 1, (total,), generator=generator).tolist()
    prepared = []
    cursor = 0
    for template_index, row in enumerate(rows):
        neutral_prefix = backend.prefix(row["neutral"], row["lyrics"])
        positive_prefix = backend.prefix(row["positive"], row["lyrics"])
        for _ in range(int(sample_seeds)):
            sample_seed = int(drawn[cursor])
            cursor += 1
            history = (
                backend.continuation(neutral_prefix, int(history_tokens), sample_seed)
                if int(history_tokens) > 0
                else []
            )
            if max(len(neutral_prefix), len(positive_prefix)) + len(history) > limit:
                raise ValueError("Prompt plus history exceeds the YuE2 context limit")
            neutral_ids = neutral_prefix + history
            positive_ids = positive_prefix + history
            with torch.no_grad():
                neutral = backend.hidden(neutral_ids)[:, -1].float()
                positive = backend.hidden(positive_ids)[:, -1].float()
            prepared.append(
                {
                    "train_ids": neutral_ids,
                    "neutral": neutral.cpu(),
                    "targets": positive.cpu(),
                    "template_row": template_index,
                    "sample_seed": sample_seed,
                }
            )
    if len(prepared) != total:
        raise RuntimeError("Seed bank length drifted")
    return prepared


def train_step(backend, network, critic, g_opt, d_opt, p_opt, rows, stamp, step: int) -> dict:
    device = network.particles.device
    neutrals = torch.cat([row["neutral"].to(device) for row in rows])
    targets = torch.cat([row["targets"].to(device) for row in rows])
    d_loss, g_loss, vic = stamp.losses()
    regularizer = stamp.regularizer()
    sigma = float(stamp.noise_std_at(step, float(critic.edit_rms)))
    real = critic.normalize((targets - neutrals).detach())
    noise = torch.randn_like(real) * sigma

    def student() -> torch.Tensor:
        pieces = []
        with network.scaled(1.0):
            for row in rows:
                pieces.append(backend.hidden(row["train_ids"])[:, -1].float())
        return torch.cat(pieces)

    critic.requires_grad_(True)
    d_opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = critic.normalize(student() - neutrals)
    real_in = (real + noise).detach().requires_grad_(True)
    fake_in = (fake + noise).detach().requires_grad_(True)
    penalty, stats = regularizer.penalty(critic, real_in, fake_in, step=step)
    d_total = d_loss(critic(real_in), critic(fake_in)) + penalty
    if not torch.isfinite(d_total):
        raise FloatingPointError("Non-finite YuE2 discriminator loss")
    d_total.backward()
    d_opt.step()

    critic.requires_grad_(False)
    g_opt.zero_grad(set_to_none=True)
    p_opt.zero_grad(set_to_none=True)
    fake_g = critic.normalize(student() - neutrals) + noise.detach()
    adv = g_loss(critic(real_in.detach()), critic(fake_g))
    vic_batch = int(stamp.spec["particle_vic_batch"])
    vic_loss = vic(network.particles[:vic_batch])
    g_total = float(stamp.spec["adv_weight"]) * adv + float(stamp.spec["vicreg_weight"]) * vic_loss
    if not torch.isfinite(g_total):
        raise FloatingPointError("Non-finite YuE2 generator loss")
    g_total.backward()
    g_opt.step()
    p_opt.step()
    return {
        "step": step,
        "d_loss": float(d_total.detach()),
        "g_loss": float(g_total.detach()),
        "g_adv": float(adv.detach()),
        "vic": float(vic_loss.detach()),
        "penalty": float(penalty.detach()),
        "sigma": sigma,
        "penalty_applied": bool(stats.get("applied", False)),
    }


def _ema(shadow: dict[str, torch.Tensor], network: ParticleSlider, decay: float) -> None:
    with torch.no_grad():
        for key, value in network.state_dict().items():
            shadow[key].mul_(decay).add_(value.detach().float().cpu(), alpha=1.0 - decay)


def train(args: argparse.Namespace) -> dict | Path:
    prompts_path = _resolve_path(args.prompts_file)
    rows, _meta = load_prompts(prompts_path)
    stamp, locked = resolve_surface(args, len(rows))
    assert_product_game(stamp)
    card = card_for(args, stamp, locked, rows=len(rows))
    if args.print_card:
        print(json.dumps(card, indent=2))
        return card
    if args.dummy:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    backend = YuE2Backend(
        args.model_id,
        vae_id=args.vae_id,
        device=args.device,
        allow_hub=args.allow_hub,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dummy=args.dummy,
    )
    prepared = prepare_bank(
        backend,
        rows,
        sample_seeds=int(locked["sample_seeds"]),
        history_tokens=int(locked["history_tokens"]),
        seed=args.seed,
    )
    network = ParticleSlider(backend.model, stamp)
    device = next(backend.model.parameters()).device
    targets = torch.cat([row["targets"] for row in prepared]).to(device)
    neutrals = torch.cat([row["neutral"] for row in prepared]).to(device)
    critic = stamp.critic(targets, neutrals=neutrals)
    betas = tuple(locked["betas"])
    adapter_params = [param for name, param in network.named_parameters() if name != "particles"]
    g_opt = torch.optim.Adam(adapter_params, lr=float(locked["g_lr"]), betas=betas)
    p_opt = torch.optim.Adam([network.particles], lr=float(locked["particle_lr"]), betas=betas)
    d_opt = torch.optim.Adam(critic.parameters(), lr=float(locked["d_lr"]), betas=betas)
    shadow = {key: value.detach().float().cpu().clone() for key, value in network.state_dict().items()}
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        card,
        prompts_file=str(prompts_path),
        seed=args.seed,
        dummy=bool(args.dummy),
        validation_status="unvalidated",
        recommended_range=list(locked["recommended_range"]),
    )
    log_path = args.save_dir / f"{args.name}_train.jsonl"
    started = time.monotonic()
    metrics: dict = {}
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            start = ((step - 1) * int(locked["adv_batch"])) % len(prepared)
            batch = [prepared[(start + offset) % len(prepared)] for offset in range(int(locked["adv_batch"]))]
            metrics = train_step(backend, network, critic, g_opt, d_opt, p_opt, batch, stamp, step)
            _ema(shadow, network, float(locked["ema"]))
            metrics["elapsed_sec"] = time.monotonic() - started
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if step == 1 or step == args.steps or step % 10 == 0:
                print(json.dumps(metrics), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                network.save(
                    args.save_dir / f"{args.name}_{step}.safetensors",
                    dict(metadata, **metrics),
                    state=shadow,
                )
    last = args.save_dir / f"{args.name}_last.safetensors"
    network.save(last, dict(metadata, **metrics), state=shadow)
    sidecar = args.save_dir / f"{args.name}.json"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None):
    return train(parse_args(argv))
