"""YuE2 product constants. Formulation numbers are not stored here."""

from __future__ import annotations

MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
UPSTREAM_REPO = "https://github.com/multimodal-art-projection/YuE"
UPSTREAM_REVISION = "ef1936f2ee39fe8de486a0f47a481c95f8d4da87"
HUB_PROJECT = "ntc-ai/yue2-concept-sliders"
HUB_SPACE = "spaces/ntc-ai/yue2-concept-sliders"
WEIGHTS_TREE = "weights/particle-1200-v1"

CORE_PIN = "ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"
CORE_REPO = "https://github.com/HyperGAN/particle-sliders"

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
PRODUCT_NAME = "yue2-particle-sliders"

# Hub recipe name recorded on the provisional stamp. The value is checked
# against winning_formulation(); it is not a second copy of the stamp.
HUB_RECIPE_NAME = "anneal-routed-particle-error-yue2-v1"
