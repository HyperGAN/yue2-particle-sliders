"""ComfyUI gate for a YuE2 routed-particle slider.

Drop this repo in ``ComfyUI/custom_nodes``. The node checks a local
safetensors slider against ``winning_formulation()`` and returns a slider
handle. It does not download Hub weights and it does not synthesize audio.
Song rendering stays in ``scripts/infer_yue2.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

from yue2_sliders.defaults import MODEL_ID
from yue2_sliders.integration import configure_integration
from yue2_sliders.slider import validate_checkpoint, validate_record
from yue2_sliders.stamp import locked_stamp

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def validate_strength(strength: float, recommended: tuple[float, float] | list[float]) -> float:
    value = float(strength)
    low, high = (float(recommended[0]), float(recommended[1]))
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"Strength must be between {low:g} and {high:g}")
    return value


def validate_slider_record(record: dict, stamp) -> dict:
    validate_record(record, stamp)
    recipe = record.get("recipe_name", record.get("recipe"))
    return {
        "format": record["format"],
        "recipe_name": recipe,
        "formulation_id": stamp.formulation_id,
        "model_id": MODEL_ID,
        "rank": int(stamp.spec["adapter_rank"]),
        "particles": int(stamp.spec["parts"]),
    }


def validate_slider_file(path: str | Path, stamp=None) -> dict:
    configure_integration()
    stamp = stamp or locked_stamp()[0]
    record, _state = validate_checkpoint(path, stamp)
    checked = validate_slider_record(record, stamp)
    checked["path"] = str(path)
    return checked


class YuE2ParticleSlider:
    """Validate a local YuE2 particle slider and publish its strength."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("YUE2_SLIDER",)
    RETURN_NAMES = ("slider",)
    FUNCTION = "load"
    CATEGORY = "NTC/YuE2"

    def load(self, weights_path, strength):
        configure_integration()
        stamp, locked = locked_stamp()
        value = validate_strength(strength, locked["recommended_range"])
        if value == 0.0:
            return ({"strength": 0.0, "path": "", "recipe_name": locked["recipe_name"], "model_id": MODEL_ID},)
        path = Path(weights_path)
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("YuE2 slider must be an existing .safetensors file")
        checked = validate_slider_file(path, stamp)
        checked["strength"] = value
        return (checked,)


NODE_CLASS_MAPPINGS = {"NTCYuE2ParticleSlider": YuE2ParticleSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"NTCYuE2ParticleSlider": "YuE2 Particle Slider (ntc-ai)"}
