"""Owned eager native AR teacher/history bank; no GAN math or persistent cache."""
from copy import deepcopy

import torch


def prepare_bank(backend, adapter, rows, *, sample_seeds, history_tokens, seed):
    from particle_sliders.audio import AudioRow

    limit = int(getattr(backend.model.config, "max_position_embeddings", 10**9))
    total = len(rows) * int(sample_seeds)
    generator = torch.Generator().manual_seed(int(seed) + 12345)
    drawn = torch.randint(0, 2**31 - 1, (total,), generator=generator).tolist()
    prepared = []
    for row in rows:
        neutral = backend.prefix(row["neutral"], row["lyrics"])
        positive = backend.prefix(row["positive"], row["lyrics"])
        for _ in range(int(sample_seeds)):
            with adapter.network.scaled(0):
                history = backend.continuation(neutral, int(history_tokens), int(drawn[len(prepared)]))
            if max(len(neutral), len(positive)) + len(history) > limit:
                raise ValueError("Prompt plus history exceeds the YuE2 context limit")
            prepared.append(adapter.prepare(AudioRow(tuple(neutral + history), tuple(positive + history),
                                                    history_tokens=len(history))))
    return prepared


class PreparedBank:
    def __init__(self, runtime, adapter, rows, *, sample_seeds, history_tokens, seed):
        limit = int(runtime.model.config.max_position_embeddings)
        # Reject before continuation can allocate an oversized history tensor.
        if history_tokens >= limit:
            raise ValueError('history must leave room for a native prompt prefix')
        for row in rows:
            if max(len(runtime.prefix(row['neutral'], row['lyrics'])),
                   len(runtime.prefix(row['positive'], row['lyrics']))) + history_tokens > limit:
                raise ValueError('prompt plus history exceeds the native context limit')
        from particle_sliders.audio import audio_conditions
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
