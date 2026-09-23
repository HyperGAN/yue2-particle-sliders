"""YuE2 product constants. Formulation numbers are not stored here."""

from __future__ import annotations

MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
UPSTREAM_REPO = "https://github.com/multimodal-art-projection/YuE"
UPSTREAM_REVISION = "ef1936f2ee39fe8de486a0f47a481c95f8d4da87"
HUB_PROJECT = "ntc-ai/yue2-concept-sliders"
HUB_SPACE = "spaces/ntc-ai/yue2-concept-sliders"
WEIGHTS_TREE = "weights/particle-1200-v1"

CORE_PIN = "4340e28bed388d50800c469525b460a108091da0"
CORE_REQUIREMENT = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@"
    f"{CORE_PIN}#subdirectory=packages/particle-sliders-core"
)
CORE_REPO = "https://github.com/HyperGAN/particle-sliders"
# Parameter table for the provisional overlay. It stays in particle-sliders.
EXAM_MODULE = "analysis/slider2d/yue2_gmix_v2_exam.py"

# Checkpoint formats this product reads. New exports use PRODUCT_FORMAT.
PRODUCT_FORMAT = "yue2-particle-sliders-routed-ar-v1"
PUBLISHED_FORMAT = "conceptmod-yue2-routed-particle-ar-v1"
FORMATS = (PRODUCT_FORMAT, PUBLISHED_FORMAT)

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
PRODUCT_NAME = "yue2-particle-sliders"

# Hub recipe name recorded on the provisional stamp. The value is checked
# against winning_formulation(); it is not a second copy of the stamp.
HUB_RECIPE_NAME = "anneal-routed-particle-error-yue2-v1"
