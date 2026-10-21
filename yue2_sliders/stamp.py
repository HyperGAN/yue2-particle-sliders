"""Lock routed adapter geometry to the shared winning formulation.

The stamp math lives in particle-sliders-core. This module only calls
``winning_formulation()`` and ``require()``.
"""

from __future__ import annotations

from yue2_sliders.defaults import CORE_PIN, HUB_RECIPE_NAME


def locked_stamp(model_surface: dict | None = None):
    """Return the shared stamp after ``require()`` accepts the product card.

    ``model_surface`` may set learning rates, batch size and data-budget
    fields. Any formulation key, provenance fork, or unknown knob raises.
    """
    from particle_sliders import winning_formulation

    stamp = winning_formulation()
    if stamp.spec.get("recipe_name") != HUB_RECIPE_NAME:
        raise RuntimeError(
            f"Core pin {CORE_PIN} recipe_name is {stamp.spec.get('recipe_name')!r}, "
            f"expected the provisional Hub recipe {HUB_RECIPE_NAME!r}. "
            "Bump the pin; do not fork the stamp in this repo."
        )
    card = stamp.as_dict()
    surface = dict(model_surface or {})
    unknown = set(surface) - set(stamp.model_surface_keys)
    if unknown:
        raise ValueError(
            f"Refusing formulation overrides {sorted(unknown)}. "
            "Change the stamp in HyperGAN/particle-sliders."
        )
    card.update(surface)
    stamp.require(card)
    return stamp, card
