"""Lock this product to the shared winning formulation.

The stamp math lives in particle-sliders-core. This module only calls
``winning_formulation()`` and ``require()``.
"""

from __future__ import annotations

from particle_sliders import winning_formulation

from yue2.defaults import CORE_PIN, HUB_RECIPE_NAME


def locked_stamp(model_surface: dict | None = None):
    """Return the shared stamp after ``require()`` accepts the product card.

    ``model_surface`` may set learning rates, batch size and data-budget
    fields. Any formulation key, provenance fork, or unknown knob raises.
    """
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


def assert_product_game(stamp) -> None:
    """This train loop only runs the stamped gmix paired-error game.

    Nonzero auxiliary weights are not reimplemented here. A future overlay
    that turns them on has to land in the shared core first.
    """
    if stamp.architecture_id != "gmix" or stamp.spec["critic"] != "gmix":
        raise RuntimeError("YuE2 training expects the gmix architecture from winning_formulation()")
    if stamp.spec["generator_objective"] != "paired_error_rpgan_plus_particle_vic":
        raise RuntimeError("YuE2 training expects the stamped paired-error particle game")
    aux = stamp.spec["aux_weights"]
    active = sorted(key for key, value in aux.items() if float(value) != 0.0)
    if active:
        raise RuntimeError(
            f"Nonzero aux weights {active} are not a YuE2 product fork. "
            "Teach them in particle-sliders-core."
        )
