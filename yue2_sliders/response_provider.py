"""Owned eager native AR teacher/history bank; no GAN math or persistent cache."""
from copy import deepcopy


class PreparedBank:
    def __init__(self, runtime, adapter, rows, *, sample_seeds, history_tokens, seed):
        # Legacy preparation preserves exact tokenization/shared-history behavior.
        # Imported only for training construction, never by native-only inference.
        limit = int(runtime.model.config.max_position_embeddings)
        # The historical helper checks context after continuation; reject here
        # before that helper can allocate an oversized history tensor.
        if history_tokens >= limit:
            raise ValueError('history must leave room for a native prompt prefix')
        for row in rows:
            if max(len(runtime.prefix(row['neutral'], row['lyrics'])),
                   len(runtime.prefix(row['positive'], row['lyrics']))) + history_tokens > limit:
                raise ValueError('prompt plus history exceeds the native context limit')
        from .train import prepare_bank
        from particle_sliders.pg09_audio import audio_conditions
        self._rows = tuple(prepare_bank(runtime, adapter, rows, sample_seeds=sample_seeds,
                                      history_tokens=history_tokens, seed=seed))
        self._conditions = audio_conditions(self._rows).detach().clone()
        self.response_shape = tuple(self._rows[0].target.shape)
        self.closed = False

    def _open(self):
        if self.closed:
            raise RuntimeError('YuE2 prepared response bank is closed')

    def __len__(self):
        self._open()
        return len(self._rows)

    def __getitem__(self, index):
        self._open()
        if type(index) is not int or not 0 <= index < len(self._rows):
            raise IndexError('YuE2 response bank index out of bounds')
        return deepcopy(self._rows[index])

    def conditions(self, indices):
        self._open()
        return self._conditions[indices.to(self._conditions.device)].detach().clone()

    def identity(self):
        self._open()
        return dict(protocol='yue2-owned-prepared-ar-bank-v1', rows=deepcopy(self._rows),
                    conditions=self._conditions.detach().clone())

    @property
    def condition_width(self):
        self._open()
        return self._conditions.shape[1]

    def close(self):
        self.closed = True
        self._rows = ()
        self._conditions = None
