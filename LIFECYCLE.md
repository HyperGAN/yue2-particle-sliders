# YuE2 neutral numerical response proposal

The isolated proposal uses **core0.8.2 / PG0.9.1**, exact shared source
`21761ca8fd059439b58aa56a69bf8d1647f50e31`. PG0.9.1 is the existing PR4 cohort's
version; this is a new product source cohort, without a PG downgrade or an old
checkpoint migration. Historical CDF locks, CLI packet formats, receipts, PR3's
unmerged retraining payload, and active environments remain untouched.

Shared contract review is parent-supplied evidence: 71 installed-runtime checks
per dependency line, independently reviewed, with exact canonical source maps.
**This product proposal has not been built, installed, imported or tested.**
Only source parsing, archive/source hashing and Git whitespace checks ran here.
The [audit](docs/lifecycle-integration-audit.md) and
[pending test plan](docs/response-session-test-plan.md) distinguish that boundary.

`yue2_sliders.lifecycle:BACKEND` now registers exclusively in
`particle_sliders.response_backends`. Its descriptor is explicit:
`modality=audio_ar_hidden`, `response_kind=native_final_ar_hidden`,
`response_domain=continuous_vector`, native shape `[1,32]` for the CPU fixture
and `[1,2048]` for the pretrained product, actual `routed_particle_gan`, exact
PG0.9.1 `release/auto`. Response-session packets are separate from image packets.
Rendering, image references, negative strengths and waveform/token output are
unavailable. The original AR hidden state is never projected, truncated,
spatially reshaped or treated as a categorical token response.

The complete factory owns the existing frozen YuE2 loader, real 128x4 routed
bank, rank8/alpha8 q/k/v/o projection hooks, native positive/neutral histories,
shared AR response adapter, eager immutable teacher/condition provider, existing
raw-logit audio critic and ConditionalPG09. Shared PG owns GAN math, optimizers,
coverage and RNG. HyperGAN owns run presentation via `on_step`/manifest/artifacts.
Native decoder/backbone, routing, conditioning and network laws are unchanged;
HNDL conversion remains separate matched-weight work.

The provider owns its copied teacher rows and full neutral-response conditions,
returns copies and releases them on close. Observation covers provider bounds/fetch
and generated responses through the reviewed common scope, plus product-owned
native buffer registrations/values, modes, gradient/checkpoint flags and forwards.
Setup failures and close restore projection/checkpoint hooks. Frozen base weights
are immutable; observation does not clone the whole pretrained backbone.

After a parent-admitted build/install/CPU slot, the intended API is:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig('yue2', 'new-yue2-response-cohort', dummy=True,
    rows=[dict(neutral='a quiet acoustic song', positive='a loud guitar song',
               lyrics='[verse]\nA train rolls through the night'),
          dict(neutral='a soft piano song', positive='a bright brass song',
               lyrics='[verse]\nWe meet beside the river')],
    steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint('new-run/training.pt')
with open_response(config, resume='new-run/training.pt') as run:
    run.train()
    run.export('new-run/adapter.pt', selection='ema')
    run.export_inference('new-run/adapter-v2.pt', selection='ema')
    hidden_response = run.evaluate(index=0, seed=11, strength=.5)
```

Native-only v2 inference is a distinct installed registration:
`yue2_sliders.response_native:INFERENCE` in `particle_sliders.inference_backends`.
It builds only the frozen base, actual slider and shared AR response adapter,
without importing the product training factory, provider or train module. It
constructs no teacher data, critic, game or optimizer. The export binds actual
base identity/digest, adapter schema and implementation sources, protocol,
architecture/context limit, fixture base seed, precision and exact PG profile.
It contains neither paired prompt rows nor teacher/condition cache values.

```python
from particle_sliders.inference import open_inference

with open_inference('new-run/adapter-v2.pt', options={'device': 'cpu'},
                    operation='response') as native:
    hidden_response = native.response(token_ids=[12, 21, 30], strength=.5, seed=11)
```

`token_ids` must be an existing native AR prefix/history, nonempty, in vocabulary
and within the context limit. The callback evaluates its final hidden state; it
never samples tokens or renders a song. Native runtime JSON accepts only `device`,
`revision` and `cache_dir`; it does not accept training configuration/data. Dummy
mode needs only the optional CPU device; its frozen base is reconstructed from
the artifact's bound seed. Native BF16 would require an explicit CUDA device and
separate admission, but native width2048 currently fails the PG source/shape gate
before any pretrained allocation. A new native prompt's tokenizer/history and
acoustic qualification remain separate work.

CLI equivalents use `--interface response`, `--export-format both`, and
`particle-slider infer --mode response` with native JSON `{"token_ids":[12,21,30]}`.
See the explicit commands in the test plan. Raw/EMA receipt selection is retained;
an inference artifact cannot train or resume.

The packaged `lifecycle-runtime.lock.json` is an **unchanged copy** of the owner's
canonical 091 lock fragment, including all 38 source hashes and audio closure.
Core wheel SHA256 is
`d28d082986accc486a4a9c2ce445f4981daaba6f51605b20ded4c0f424cb30ba`;
companion `reviewed-source.tar` SHA256 is
`f90b93d4a402b0fbe499d9335b6243f20ab64229c2c0e0703da011743ccd30eb`.
Both were hash-verified read-only. The three guides omitted from the sdist were
read from that verified companion and checked against the exact source checkout.
The owner artifacts' older pending-status fields remain unchanged; the parent's
latest delegation supplies independent-review completion. No product wheel hash
or product installed-pass receipt is claimed.

Runtime verification requires exact canonical source bytes, metadata version,
exact PG dependency and every released PG source. A recorded direct archive hash
must match. The owner's reviewed isolated wheel-extraction layout is also allowed
when direct_url metadata is absent, with the same mandatory source checks. No
checkout fallback, path injection, model download or environment mutation occurs.

Only released `release/auto` is selected in this proposal. The old optional
`alz-continuous-v1` source is unrecovered and is explicitly rejected. Both official
PG0.9.1 and PG0.9.3 lack the admitted wide continuous selector; native width2048
stays gated. This is not a preflight API defect or permission to reshape data.

Legacy CLI/Comfy still select their original core0.6 CDF lock and active-weight
schemas. Use the original PR4 source checkout for its original source-bound
resumes. The earlier namespace fix follows PR3's `yue2_sliders` separation while
native runtime imports remain `yue2`; no legacy packet is rebound to this new
response or native-only contract.
