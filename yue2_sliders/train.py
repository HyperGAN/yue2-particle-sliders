"""Train a YuE2 AR slider on the shared winning formulation.

``python scripts/train_yue2.py --dummy`` runs on CPU and does not download Hub
weights. The shared ``ConditionalPG09`` owns every update, optimizer and checkpoint.
``winning_formulation()`` supplies only the existing routed adapter architecture.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import hashlib
import time
from pathlib import Path

import torch
import yaml

from yue2_sliders.backend import YuE2Backend
from yue2_sliders.defaults import CORE_PIN, EXAM_MODULE, MODEL_ID, VAE_ID
from yue2_sliders.prompts import load_prompts
from yue2_sliders.slider import ParticleSlider
from yue2_sliders.stamp import locked_stamp
from yue2_sliders.integration import configure_integration

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
    "preset",
    "sampling",
    "pg_source_profile",
    "penalty_input_domain",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train a routed-particle slider on m-a-p/YuE2-3B. "
            "ConditionalPG09 trains the public GAN-only recipe; winning_formulation() supplies adapter geometry. "
            "--dummy does not download Hub weights."
        ),
    )
    parser.add_argument("--config_file", type=Path, default=ROOT / "configs" / "yue2" / "config-yue2-metal.yaml")
    parser.add_argument("--resume", type=Path, help="Full training checkpoint; inference weights cannot resume")
    parser.add_argument("--until", type=int, help="Pause after this update, retaining the --steps recipe horizon")
    parser.add_argument("--preset", choices=("bispf", "bispf_optimized"), default="bispf_optimized")
    parser.add_argument("--sampling", choices=("balanced", "replacement"), default="balanced")
    parser.add_argument("--pg_source_profile", choices=("release", "alz-continuous-v1"), default="release")
    parser.add_argument("--penalty_input_domain", choices=("auto", "continuous"), default="auto")
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
    dependency = configure_integration()
    if initial.config_file and not Path(initial.config_file).is_file():
        parser.error("config_file must exist")
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
    args.dependency = dependency
    args.explicit_steps = _flag_set(sys.argv[1:] if argv is None else argv, "--steps")
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
    if args.until is not None and not 1 <= args.until <= args.steps:
        parser.error("until must be in [1, steps]; steps is the fixed recipe horizon")
    for key in ("g_lr", "d_lr", "particle_lr"):
        value = getattr(args, key)
        if value is not None:
            parser.error(f"{key} is a legacy game knob; the public PG09 preset owns optimizer rates")
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
    if args.until is not None and args.until > args.steps:
        raise ValueError("until exceeds the resolved recipe horizon")
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
        "training_game": "conditional-pg09-v1",
        "training_recipe": args.preset,
        "sampling": args.sampling,
        "trained_scales": [1.0],
        "negative_strength_qualified": False,
        "integration_dependencies": args.dependency,
        "exam_module": EXAM_MODULE,
        "rows": rows,
        "steps": args.steps,
        "dummy": bool(args.dummy),
        "allow_hub": bool(args.allow_hub),
        "model_surface": {key: locked[key] for key in stamp.model_surface_keys
                          if key not in ("g_lr", "d_lr", "particle_lr")},
        "non_goals": [
            "does not vendor RoutedMLP, GradRegularizer, locked_shared, or ParticleGAN",
            "does not fork the provisional exam; it stays in particle-sliders",
            "does not download Hub weights in --dummy",
        ],
    }


def prepare_bank(backend, adapter, rows, *, sample_seeds, history_tokens, seed):
    from particle_sliders.pg09_audio import AudioRow

    limit = int(getattr(backend.model.config, "max_position_embeddings", 10**9))
    total = len(rows) * int(sample_seeds)
    generator = torch.Generator().manual_seed(int(seed) + 12345)
    drawn = torch.randint(0, 2**31 - 1, (total,), generator=generator).tolist()
    prepared = []
    for row in rows:
        neutral = backend.prefix(row["neutral"], row["lyrics"])
        positive = backend.prefix(row["positive"], row["lyrics"])
        for _ in range(int(sample_seeds)):
            with adapter.network.scaled(0):
                history = backend.continuation(neutral, int(history_tokens), int(drawn[len(prepared)]))
            if max(len(neutral), len(positive)) + len(history) > limit:
                raise ValueError("Prompt plus history exceeds the YuE2 context limit")
            prepared.append(adapter.prepare(AudioRow(tuple(neutral + history), tuple(positive + history),
                                                    history_tokens=len(history))))
    return prepared


def _frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_identity():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def _atomic_save(packet, path):
    pending = path.with_suffix(".pending")
    torch.save(packet, pending)
    pending.replace(path)


def train(args: argparse.Namespace) -> dict | Path:
    from particle_sliders import ConditionalPG09
    from particle_sliders.pg09_audio import (
        YuE2ARResponseAdapter, AudioResponseCritic, audio_conditions, require_audio_width,
    )

    prompts_path = _resolve_path(args.prompts_file)
    rows, _meta = load_prompts(prompts_path)
    stamp, locked = resolve_surface(args, len(rows))
    card = card_for(args, stamp, locked, rows=len(rows))
    if args.print_card:
        print(json.dumps(card, indent=2))
        return card
    options = dict(pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
    # Validate the public penalty domain before loading any pretrained tensors.
    if args.dummy:
        width = 32
    else:
        from yue2_sliders.backend import require_yue2
        require_yue2()
        from yue2.storage import resolve_model
        resolved = resolve_model(args.model_id, revision=args.revision,
                                 cache_dir=args.cache_dir, local_files_only=not args.allow_hub)
        width = json.loads((Path(resolved) / "config.json").read_text())["hidden_size"]
    require_audio_width(width, **options)
    if args.save_dir.exists() and (not args.save_dir.is_dir() or any(args.save_dir.iterdir())):
        raise FileExistsError("Use a fresh save directory; resume writes to a separate continuation directory")
    if args.dummy:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    backend = YuE2Backend(args.model_id, vae_id=args.vae_id, device=args.device,
        allow_hub=args.allow_hub, revision=args.revision, cache_dir=args.cache_dir, dummy=args.dummy)
    frozen = _frozen_digest(backend.model)
    network = ParticleSlider(backend.model, stamp)
    adapter = YuE2ARResponseAdapter(backend, network, model_identity=dict(
        base=backend.identity, product_sources=_source_identity(),
        prompts_sha256=hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        integration_dependencies=args.dependency),
        checkpoint_modules=[] if args.dummy else list(backend.model.model.layers), **options)
    prepared = prepare_bank(backend, adapter, rows, sample_seeds=locked["sample_seeds"],
                            history_tokens=locked["history_tokens"], seed=args.seed)
    conditions = audio_conditions(prepared)
    device = next(network.parameters()).device
    critic = AudioResponseCritic(width, conditions.shape[1]).to(device)
    game = ConditionalPG09(adapter, prepared, critic, conditions=conditions,
        particles=adapter.particles, response_domain="continuous_vector",
        preset=args.preset, batch_size=int(locked["adv_batch"]), seed=args.seed,
        total_steps=args.steps, sampling=args.sampling, **options)
    metrics = []
    if args.resume:
        packet = torch.load(args.resume, map_location="cpu", weights_only=True)
        if packet.get("frozen_base_sha256") != frozen:
            raise ValueError("Resume checkpoint frozen base differs")
        game.load_state_dict(packet["game"])
        metrics = packet["metrics"]
        if len(metrics) != game.completed_steps:
            raise ValueError("Resume metrics differ from completed steps")
    until = args.steps if args.until is None else args.until
    if until <= game.completed_steps:
        raise ValueError("until must exceed the completed checkpoint step")
    metadata = dict(card, model_identity=backend.identity, seed=args.seed,
        prompts_file=str(prompts_path), dummy=bool(args.dummy), validation_status="unvalidated",
        recommended_range=list(locked["recommended_range"]), game_signature=game.state_dict()["config"],
        training_weights="active", ema=False, audio_rendered=False, quality_qualified=False)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / f"{args.name}_train.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for row in metrics:
            log.write(json.dumps(row) + "\n")
        while game.completed_steps < until:
            row = game.step()
            metrics.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            step = game.completed_steps
            print(json.dumps(dict(row, elapsed_sec=time.monotonic() - started)), flush=True)
            if step % args.save_every == 0 or step == until:
                _atomic_save(dict(game=game.state_dict(), frozen_base_sha256=frozen, metrics=metrics,
                                  probes=[game.evaluate(i, seed=1709) for i in range(min(2, len(prepared)))]),
                             args.save_dir / f"{args.name}_checkpoint.pt")
                network.save(args.save_dir / f"{args.name}_{step}.safetensors",
                             dict(metadata, completed_steps=step))
    if _frozen_digest(backend.model) != frozen:
        raise ValueError("Frozen YuE2 base changed during training")
    network.save(args.save_dir / f"{args.name}_last.safetensors",
                 dict(metadata, completed_steps=game.completed_steps))
    _atomic_save(game.export_state_dict(), args.save_dir / f"{args.name}_adapter-export.pt")
    sidecar = args.save_dir / f"{args.name}.json"
    sidecar.write_text(json.dumps(dict(metadata, completed_steps=game.completed_steps), indent=2) + "\n",
                       encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None):
    return train(parse_args(argv))
