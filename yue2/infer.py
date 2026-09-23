"""Render a matched YuE2 scale comparison.

``--dummy`` checks a CPU checkpoint at several strengths and writes JSON.
It does not synthesize audio and does not download weights. A live run uses
the official YuE2 eager pipeline with the adapter on AR only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from yue2.backend import TinyYuE2, YuE2Backend, file_digest, require_yue2, tiny_from_architecture
from yue2.defaults import CORE_PIN, MODEL_ID, VAE_ID
from yue2.prompts import sound_only
from yue2.slider import ParticleSlider, read_record
from yue2.stamp import locked_stamp

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare YuE2 slider strengths on m-a-p/YuE2-3B. "
            "One neutral caption, lyric sheet and seed. "
            "The comparison is stamped with winning_formulation(). "
            "--dummy does not download Hub weights."
        ),
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--style", default=None, help="Neutral sound description")
    parser.add_argument("--lyrics_file", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--vae_id", default=VAE_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--vae_revision", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory_budget_gib", type=float, default=20)
    parser.add_argument("--scales", default="0,0.5,1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cot", choices=["off", "melody", "full"], default="off")
    parser.add_argument("--abc_file", type=Path)
    parser.add_argument("--max_tokens", type=int, default=9000)
    parser.add_argument("--allow_hub", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--print_card", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.scales = [float(item) for item in str(args.scales).split(",") if item != ""]
    except ValueError:
        parser.error("scales must be comma-separated numbers")
    if not args.scales or any(not math.isfinite(scale) for scale in args.scales):
        parser.error("scales must be finite")
    if len(set(args.scales)) != len(args.scales):
        parser.error("scales must be unique")
    if args.model_id != MODEL_ID or args.vae_id != VAE_ID:
        parser.error(f"model_id must be {MODEL_ID} and vae_id must be {VAE_ID}")
    if args.max_tokens < 1 or not 0 <= args.seed < 2**63:
        parser.error("max_tokens must be positive; seed must be in [0, 2**63)")
    if not args.print_card and (args.weights is None or args.output_dir is None):
        parser.error("--weights and --output_dir are required")
    if not args.dummy and not args.print_card and (args.style is None or args.lyrics_file is None):
        parser.error("a live render needs --style and --lyrics_file")
    return args


def comparison_card(args: argparse.Namespace) -> dict:
    stamp, locked = locked_stamp()
    return {
        "backend": "yue2",
        "product": "yue2-particle-sliders",
        "model_id": args.model_id,
        "vae_id": args.vae_id,
        "architecture_id": stamp.architecture_id,
        "formulation_id": stamp.formulation_id,
        "formulation_provisional": stamp.formulation_provisional,
        "recipe_name": locked["recipe_name"],
        "recommended_range": list(locked["recommended_range"]),
        "scales": args.scales,
        "seed": args.seed,
        "dummy": bool(args.dummy),
        "allow_hub": bool(args.allow_hub),
        "core_pin": CORE_PIN,
    }


def _dummy_compare(args: argparse.Namespace, stamp) -> dict:
    record = read_record(args.weights)
    if not record.get("dummy"):
        raise ValueError("--dummy inference only reloads a --dummy training export")
    model = tiny_from_architecture(record["architecture"]) if record.get("architecture") else TinyYuE2()
    network, loaded = ParticleSlider.load(model, args.weights, stamp, allow_dummy=True)
    backend = YuE2Backend(dummy=True)
    backend.model = model
    style = args.style or "English, dry close vocal, soft room."
    lyrics = "[Verse]\nThe rain falls on the step\n"
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
    ids = backend.prefix(sound_only(style), sound_only(lyrics))
    rows = []
    baseline = None
    with torch.no_grad():
        for scale in args.scales:
            with network.scaled(scale):
                hidden = backend.hidden(ids)[:, -1].float()
            if scale == 0.0:
                baseline = hidden
            delta = 0.0 if baseline is None else float((hidden - baseline).pow(2).mean().sqrt())
            rows.append({"scale": scale, "rms": float(hidden.pow(2).mean().sqrt()), "delta_from_off": delta})
    if baseline is not None and any(abs(scale) < 1e-8 for scale in args.scales):
        with torch.no_grad(), network.scaled(0.0):
            again = backend.hidden(ids)[:, -1].float()
        if not torch.equal(again, baseline):
            raise RuntimeError("Scale 0 did not restore the frozen YuE2 forward")
    return {
        "loaded_format": loaded.get("format"),
        "dummy": True,
        "audio": False,
        "weights_sha256": file_digest(args.weights),
        "scales": rows,
    }


def render_live(pipe, network, *, style, lyrics, scale, seed, cot, abc, semantic_sampling, adapter_identity):
    """AR planning and semantic tokens see the slider. NAR and VAE stay at scale 0."""
    from yue2.pipeline import SongResult
    from yue2.protocol import SongRequest
    from yue2.storage import identity

    if pipe.backend != "torch-eager" or pipe.quantization != "none" or pipe.offload_ar:
        raise ValueError("Slider rendering requires torch-eager, no quantization and no AR offload")
    request = SongRequest(
        style=sound_only(style),
        lyrics=sound_only(lyrics),
        seed=seed,
        cot=cot,
        abc=sound_only(abc) if abc else None,
    )
    start = time.perf_counter()
    with network.scaled(scale):
        plan = pipe.plan(request=request)
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling)
    with network.scaled(0):
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic)
        nar_seconds = time.perf_counter() - nar_start
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
    config = pipe.effective_config(request, semantic_sampling=semantic_sampling)
    config["yue2_particle_sliders"] = {
        "adapter": adapter_identity,
        "scale": scale,
        "stages": ["plan", "semantic"],
        "acoustic_scale": 0,
    }
    timing = {
        "abc": plan.timing,
        "semantic": semantic.timing,
        "nar_seconds": nar_seconds,
        "vae_seconds": time.perf_counter() - vae_start,
        "e2e_seconds": time.perf_counter() - start,
    }
    request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
    return SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)


def infer(args: argparse.Namespace) -> dict:
    stamp, _locked = locked_stamp()
    card = comparison_card(args)
    if args.print_card:
        print(json.dumps(card, indent=2))
        return card
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh output directory for this comparison")
    if args.dummy:
        body = _dummy_compare(args, stamp)
        output.mkdir(parents=True)
        payload = dict(card, **body)
        (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "dummy": True, "scales": body["scales"]}), flush=True)
        return payload
    require_yue2()
    from yue2 import YuE2Pipeline

    lyrics = sound_only(Path(args.lyrics_file).read_text(encoding="utf-8"))
    style = sound_only(args.style)
    abc = sound_only(args.abc_file.read_text(encoding="utf-8")) if args.abc_file else None
    if not lyrics.strip() or not style.strip():
        raise ValueError("style and lyrics must be nonempty")
    digest = file_digest(args.weights)
    destinations = [output / f"scale_{scale:g}" for scale in args.scales]
    with YuE2Pipeline.from_pretrained(
        args.model_id,
        vae=args.vae_id,
        revision=args.revision,
        vae_revision=args.vae_revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_hub,
        device=args.device,
        backend="torch-eager",
        memory_budget_gib=args.memory_budget_gib,
        quantization="none",
        offload_ar=False,
    ) as pipe:
        model = pipe._load_model()
        network, record = ParticleSlider.load(model, args.weights, stamp)
        identity = record.get("model_identity")
        mot = pipe.weights.get("mot")
        if identity and mot and identity != mot:
            raise ValueError("Slider was trained on different base model weights")
        output.mkdir(parents=True)
        for scale, destination in zip(args.scales, destinations):
            result = render_live(
                pipe,
                network,
                style=style,
                lyrics=lyrics,
                scale=scale,
                seed=args.seed,
                cot=args.cot,
                abc=abc,
                adapter_identity=digest,
                semantic_sampling={"max_tokens": args.max_tokens, "min_tokens": min(200, args.max_tokens)},
            )
            result.save_artifacts(destination)
            print(json.dumps({"output": str(destination), "scale": scale, "truncated": result.truncated}), flush=True)
    (output / "card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def main(argv: list[str] | None = None):
    return infer(parse_args(argv))
