"""ComfyUI custom-node entry for yue2-particle-sliders."""

try:
    from .scripts._bootstrap import bootstrap
except ImportError:
    # A checkout can be collected as a top-level module (e.g. by pytest).
    # Resolve our own helper without accidentally importing another installed
    # package named ``scripts``.
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    _spec = spec_from_file_location("_yue2_product_bootstrap", Path(__file__).parent / "scripts" / "_bootstrap.py")
    _bootstrap = module_from_spec(_spec)
    _spec.loader.exec_module(_bootstrap)
    bootstrap = _bootstrap.bootstrap

bootstrap()

if __package__:
    from .comfy_yue2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    from comfy_yue2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
