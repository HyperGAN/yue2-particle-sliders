# Narrow CDF core adoption

Current installed package candidate is
`cdfb80a76b882b4b73861b12f555263bd64ffeb7`, wheel SHA256
`4abc83988fdeb6d7478de830e7ee8ec5d9bd99c20d52f89e4ce594d50dfd39a3`.
The shared producer is [draft PR141](https://github.com/HyperGAN/particle-sliders/pull/141).
PG remains official **0.9.1**. The exact installed wheel/source map is locked in
`configs/yue2/shared-runtime.lock.json`; this product cohort is
`yue2-pg091-cdf-v1`.

The package delta from `4be` changes only `pg09.py` and `pg09_images.py`.
The YuE2 audio adapters, backends and training helpers are byte-identical.
Relevant to this consumer, started original-Adam generator clocks must equal
completed updates, while present sparse critic histories must have a clock
between one and completed updates. The Iris metadata repair is outside this
consumer's test scope; no image integration payload was added.

A new isolated `task-11/venv-pg091-cdf` installed the exact local candidate wheel
and official PG wheel. No existing environment, old checkpoint or source cohort
was retargeted. The runtime probe confirms installed `site-packages` origins,
32 matching core Python/JSON files, 49 matching released PG sources, no shared
checkout path and CUDA uninitialized. Independent review verified wheel RECORD,
all 67 archive/Git source mappings and the installed files.

Only narrow relevant checks ran:

- [Six clock cases](../evidence/pg091-cdf/clocks.xml) passed in **5.74s**. At two
  completed updates, generator clocks zero, one and three reject atomically.
  A genuinely sparse critic parameter first used on update two has clock one;
  changing it to zero or three rejects without state/output/input mutation.
  Its valid clock-one state resumes exactly through update three.
- [Ten consumer cases](../evidence/pg091-cdf/consumer.xml) passed in **17.09s**.
  They verify source-free package/Comfy origins, actual shared steps under both
  public presets, GAN-only losses, full routed-bank/frozen-base behavior,
  active weights, initial-empty and honestly unused-critic history, exact
  fresh-process 3-vs-1+2 state/metrics/probes/weights, dummy train/infer and
  actual bank/sign reload.

[Commands](../evidence/pg091-cdf/commands.sh),
[installed runtime/import/source receipt](../evidence/pg091-cdf/runtime.json),
and [exact producer build receipt](../evidence/pg091-cdf/build-receipt.json)
are committed beside the XML. These are CPU stand-in/software checks, not
pretrained native or acoustic qualification.

The previous 104-test/full and 32-test/restore receipts remain byte-for-byte
unchanged under `evidence/pg091-wheel`. They qualify core `4be` at product
`8d36956bf7e742cc001eb1cc45588ddbdee3ccc1`; they are not relabelled as CDF runs.
See the [historical wheel adoption](pg091-wheel-adoption.md). No unrelated broad
suite was rerun for this follow-up.

The YuE2 draft still targets `main`; Ada and parent control exact-head review and
merge. Existing native product `b337`, audio `9262`, shared-core `3e94` and passed
smoke lineage remain preserved. Combined native CDF execution, memory, rendering,
listening, Y20 and longer testing remain coordinator/audio-owner gates. No GPU
job, merge, model publication, service/credential/permission change or automatic
checkpoint migration occurred.
