# Reproduce YuE2 results

For new response runs, use [Installation](INSTALL-COHORTS.md) and the [response lifecycle](LIFECYCLE.md). The root package uses ParticleGAN 0.11.1 and the pinned shared core. The scoped package adds fresh initialization and the shared numerical response protocol.

Use this immutable product revision for the earlier standalone trainer, song inference, Comfy node, HNDL preview, and published weights:

```sh
git clone https://github.com/HyperGAN/yue2-particle-sliders.git
cd yue2-particle-sliders
git checkout --detach cd774d53227d341cfb11b288d5de04e6a3142e0a
```

Follow the [original reproduction guide](https://github.com/HyperGAN/yue2-particle-sliders/blob/cd774d53227d341cfb11b288d5de04e6a3142e0a/REPRODUCE.md) for its candidate wheels, source locks and `--dummy` commands. The [original HNDL preview guide](https://github.com/HyperGAN/yue2-particle-sliders/blob/cd774d53227d341cfb11b288d5de04e6a3142e0a/docs/hypergan-hndl-cohort.md) provides its independent setup. The [archived tree](https://github.com/HyperGAN/yue2-particle-sliders/tree/cd774d53227d341cfb11b288d5de04e6a3142e0a) contains the corresponding code, configurations and evidence bytes. Use the recorded source revision for each saved FULL or RAW packet.
