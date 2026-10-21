"""ComfyUI custom-node entry for yue2-particle-sliders."""

try:
    from .scripts._bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

if __package__:
    from .comfy_yue2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    from comfy_yue2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
