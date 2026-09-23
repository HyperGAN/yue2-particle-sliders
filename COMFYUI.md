# ComfyUI

`comfy_yue2.py` is the YuE2 slider gate for ComfyUI. Drop this repo in `ComfyUI/custom_nodes` and restart Comfy. The node is **YuE2 Particle Slider (ntc-ai)** in category `NTC/YuE2`.

It does not vendor the particle game, and it does not download `m-a-p/YuE2-3B`. Point `weights_path` at a local `.safetensors` slider. Strength uses the stamped range `[0, 1]`. Strength 0 needs no file and means the frozen base model. Any other strength checks the checkpoint recipe against `winning_formulation()` (`anneal-routed-particle-error-yue2-v1` on this core pin) and refuses dummy exports and foreign tensors.

The node returns a slider handle (`path`, `strength`, `recipe_name`). It does not synthesize audio inside the graph. Matched Off/On renders stay on the native YuE2 pipeline:

```bash
python scripts/infer_yue2.py \
  --weights /path/to/adapter.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales 0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

Published listening cards are on [ntc-ai/yue2-concept-sliders](https://huggingface.co/ntc-ai/yue2-concept-sliders). This node loads those particle files only after the recipe check. It does not load an ordinary PEFT LoRA under a different format.
