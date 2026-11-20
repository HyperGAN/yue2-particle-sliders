"""YuE2 autoregressive binding from the MIT-licensed core model bindings."""
from particle_sliders.audio import ARLastHiddenResponseAdapter


class YuE2ARResponseAdapter(ARLastHiddenResponseAdapter):
    family = 'yue2_native_ar'

    def __init__(self, backend, slider, **kwargs):
        self.backend = backend
        super().__init__(backend.model, slider, **kwargs)

    def _hidden(self, ids):
        # Native product selects the genuine final AR state, never logits/NAR.
        return self.backend.hidden(list(ids), checkpointing=False)[:, -1]
