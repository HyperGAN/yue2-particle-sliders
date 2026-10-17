# Reproduce

This draft uses exact local integration sources, not a public shared-package
release. Installing `requirements.txt` supplies PG **0.9.0** and CPU dependencies;
it does not provision these internal APIs. Do not pip-install the historical
core pyproject into a qualified native environment: its dependency is older.

The immutable sources are:

| Dependency | Exact commit |
|---|---|
| shared `particle-sliders-core` source | `3e94b13eb5d6cd1ceee3f018f019a68603f35696` |
| internal audio adapter source | `9262a930cd351459e93ac33cc542fa379c12fe52` |

The audio source is a local integration checkout, not a verified GitHub release.
`yue2.integration` verifies checkout heads, source bytes and imported module
origins. Supply them with `--core_source`/`--audio_source` or the environment:

```bash
export YUE2_CORE_SOURCE=/path/to/core-at-3e94b13
export YUE2_AUDIO_SOURCE=/path/to/audio-at-9262a93
export PYTHONPATH="$YUE2_CORE_SOURCE/packages/particle-sliders-core/src"
export CUDA_VISIBLE_DEVICES=''
python scripts/train_yue2.py --help
python scripts/infer_yue2.py --help
python scripts/train_yue2.py --dummy --steps 3 --until 1 --save_dir /tmp/yue2-first
python scripts/train_yue2.py --dummy --steps 3 \
  --resume /tmp/yue2-first/metal-yue2-particles_checkpoint.pt --save_dir /tmp/yue2-resumed
python scripts/infer_yue2.py --dummy \
  --weights /tmp/yue2-resumed/metal-yue2-particles_last.safetensors \
  --output_dir /tmp/yue2-cmp
python -m pytest -q
```

`--steps` is the fixed recipe horizon; `--until` pauses without changing it.
Resume uses the full `_checkpoint.pt` and a fresh or empty output directory.
`_adapter-export.pt` is the shared inference payload, and `_last.safetensors`
contains active product weights. Dummy checkpoints contain base seed/identity
and are refused for songs and Comfy. All comparisons use identical base weights.

Native runtime pin:
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87` (`yue2-infer`). The CLI loads this
installed native package first, then adds product modules. Native dependencies
must stay native: never inject a whole other environment's site-packages.

Native training requires the exact reviewed ALZ continuous-profile source
overlay and both explicit flags `--pg_source_profile alz-continuous-v1
--penalty_input_domain continuous`. The released default rejects the genuine
2048-wide AR vector before pretrained tensors load. The overlay is never applied,
installed or guessed here. GPU jobs require coordinator admission and fresh
qualification of this derivative head; this audit performs CPU/source work only.

Training calls `ConditionalPG09`; `winning_formulation()` supplies only routed
adapter geometry. Prompt sheets for sixteen historical controls remain under
`configs/yue2/catalog/`. Published historical listening samples are not evidence
of this new PG09 trainer's acoustic quality. Y20 continuation, native rendering,
listening and longer tests remain separate owner/coordinator gates.
