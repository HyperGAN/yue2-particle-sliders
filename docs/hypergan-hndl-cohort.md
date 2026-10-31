# YuE2 HyperGAN/HNDL candidate cohort

The opt-in `yue2-hndl` response and native inference factories execute real shared
networks through HyperGAN develop's HNDL builder. `HNDLSlider` selects YuE2 AR
q/k/v/o projections and calls the shared `HNDLParticleProjection`; it does not
copy its router, bridge, or particle mathematics. The audio critic is a native
vector Linear/SiLU/Linear HNDL graph with the existing raw-logit behavior.
`ConditionalPG09` and the shared response lifecycle own all loss, penalty,
optimizer, RNG, EMA, checkpoint, and inference contracts.

This is a **new source/network cohort**, with a distinct backend, protocol,
network state keys and runtime lock. Matched-weight CPU fixtures, including the actual pinned YuE2ForCausalLM at
random tiny width32 with native grouped-query attention, compare the
original product routed network's responses/bank gradients and the original
audio critic's scores/first and second derivatives. The actual native-class fixtures also run shared PG updates and exact restore
under both public presets in a separate slot matching all declared native-runtime
dependencies. The model is initialized randomly; no pretrained tensors are read.
This does not migrate an old
checkpoint, assert identical random initialization, or qualify native output.
Native projection cleanup reuses the existing conflict-aware owner. Tests cover
deferred backward after close with scale-capturing checkpoint handles.

## Exact candidate and installation boundary

| Component | Identity |
|---|---|
| Shared core | 0.12.0, joint local commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` |
| Shared core wheel SHA256 | `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad` |
| ParticleGAN | official 0.9.4, `release-0.9.4` / `auto` |
| HyperGAN | exact clean develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` |
| HNDL | exact 0.10.0 |
| Product preview | 0.3.0.dev0; Python >=3.11,<3.15 |

`hndl-runtime.lock.json` binds every shared Python/JSON/HNDL source file. The
shared `verified_builder()` verifies the actual HNDL closure, HyperGAN builder
closure and clean build provenance. The original `lifecycle-runtime.lock.json`
and historical config locks remain unchanged. Wrong sources, versions, cohorts
and profiles refuse; there is no dependency or source fallback.

The joint core commit is local and **not published**. The default product
metadata retains its supported core0.8.2/PG0.9.1 cohort. Do not install the preview
into that environment. Build the separately declared preview wheel with:

```bash
python release_tools/build_hndl_preview.py /absolute/path/to/preview-wheels
```

The generated preview metadata requests the exact joint Git commit with the
shared `hypergan-hndl` extra and registers only `yue2-hndl`. That Git requirement
cannot resolve from public GitHub until the shared owner publishes this exact
candidate (or coordinates a newly qualified exact pin). For coordinated local
qualification, install the owner-provided core wheel, exact official PG0.9.4,
HNDL0.10 and clean HyperGAN develop build into a new environment, then install the
preview wheel with `--no-deps`. Verify the declared dependencies and actual source
closures before use. No resolver override or weakened runtime verification is
provided. The builder changes only its temporary staging metadata; default
requirements and historical files are not rewritten.

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig('yue2-hndl', 'fresh-yue2-hndl-cohort', dummy=True,
    rows=[dict(neutral='a soft piano', positive='a loud guitar', lyrics='hello'),
          dict(neutral='a quiet flute', positive='a bright brass', lyrics='hello')],
    steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)
with open_response(config) as run:
    run.train()
    run.checkpoint('fresh-run/training.pt')
    run.export_inference('fresh-run/native.pt', selection='ema')
```

Native-only reload accepts existing integer token IDs and builds no teacher bank,
critic, game or optimizer. Its numerical result remains `[1,32]` for the tiny
fixture. These hidden responses are not waveform samples, generated tokens,
ordinary LoRAs or acoustic quality evidence.

## Remaining native contract

The released PG0.9.4 domain selector still classifies 2D width>=170 as categorical.
The real YuE2 hidden width is 2048. Both factories therefore refuse the native
width before constructing the pretrained loader, with no reshape, truncation,
projection, categorical substitution or gate bypass. The shared owner needs an
explicit, reviewed continuous-vector domain profile covering `[1,2048]`, its
source manifest and checkpoint identity, before this backend can admit native
YuE2. Audio/GPU qualification remains separately owned; this task has no GPU
device and ran no pretrained model, token sampler, waveform decoder or training.

Changing the exact shared source, builder, network graph, native architecture,
model weights or domain profile requires a new reviewed product cohort and
fresh checkpoints. Historical packets must run with their original source and
environment. The preview is not a package release or default-cohort upgrade.
