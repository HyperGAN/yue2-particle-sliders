"""YuE2 particle-slider product package.

Train and infer call ``particle_sliders.winning_formulation``. This package
does not vendor the routed adapter, the gradient penalty, or the V2 exam.
"""

from pkgutil import extend_path

# Direct Python imports may start with the product checkout on sys.path. Keep
# native submodules discoverable without importing or copying the runtime init.
# The CLI bootstrap imports the installed runtime first instead.
__path__ = extend_path(__path__, __name__)

from yue2.defaults import CORE_PIN, HUB_PROJECT, MODEL_ID, VAE_ID

__all__ = ["CORE_PIN", "HUB_PROJECT", "MODEL_ID", "VAE_ID"]
