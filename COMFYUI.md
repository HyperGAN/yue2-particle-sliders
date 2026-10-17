# ComfyUI

`comfy_yue2.py` is the YuE2 slider gate for ComfyUI. Drop this repo in `ComfyUI/custom_nodes` and restart Comfy. The node is **YuE2 Particle Slider (ntc-ai)** in category `NTC/YuE2`.

This integration draft needs the exact source dependencies from
[REPRODUCE.md](REPRODUCE.md). Set `YUE2_CORE_SOURCE` and `YUE2_AUDIO_SOURCE` in
the Comfy environment. The node verifies the same commits, source bytes and
module origins as the CLI before resolving the architecture stamp; it refuses
an already imported core from a different source.

It does not vendor the particle game, and it does not download `m-a-p/YuE2-3B`. Point `weights_path` at a local `.safetensors` slider. Strength uses the stamped range `[0, 1]`. Strength 0 needs no file and means the frozen base model. Any other strength checks the checkpoint recipe against `winning_formulation()` (`anneal-routed-particle-error-yue2-v1` on this core pin) and refuses dummy exports and foreign tensors.

The node returns a slider handle (`path`, `strength`, `recipe_name`). It does not synthesize audio inside the graph. Matched Off/On renders stay on the native YuE2 pipeline:

```bash
python scripts/infer_yue2.py \
  --weights /path/to/adapter.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales 0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

Published listening cards are on [ntc-ai/yue2-concept-sliders](https://huggingface.co/ntc-ai/yue2-concept-sliders). The node requires matching recorded base/stamp provenance and a complete finite tensor payload; historical files missing that provenance are refused. It does not load an ordinary PEFT LoRA under a different format.
