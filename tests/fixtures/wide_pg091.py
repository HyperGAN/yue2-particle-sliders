"""Synthetic full-width YuE2 AR response test; no pretrained weights or audio."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from yue2_sliders.integration import configure_integration
from yue2_sliders.backend import TinyYuE2, YuE2Backend, _TinyConfig
from yue2_sliders.slider import ParticleSlider
from yue2_sliders.stamp import locked_stamp
from yue2_sliders.train import _frozen_digest

configure_integration()
from particle_sliders import ConditionalPG09
from particle_sliders.pg09_audio import AudioRow, AudioResponseCritic, YuE2ARResponseAdapter, audio_conditions
from particle_sliders.pg09 import validate_pg09_domain


def equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert first == second


def build(preset):
    torch.manual_seed(7)
    backend = YuE2Backend(dummy=True)
    backend.model = TinyYuE2(_TinyConfig(hidden_size=2048, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=32, head_dim=64, vocab_size=64)).eval()
    stamp, _ = locked_stamp()
    slider = ParticleSlider(backend.model, stamp)
    options = dict(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    adapter = YuE2ARResponseAdapter(backend, slider, model_identity={'fixture': 'synthetic-2048-yue2-ar'}, **options)
    rows = [adapter.prepare(AudioRow(tuple(n), tuple(p), history_tokens=2)) for n, p in (
        ([1, 3, 5, 7], [2, 3, 5, 7]), ([8, 9, 5, 7], [10, 9, 5, 7]))]
    conditions = audio_conditions(rows)
    critic = AudioResponseCritic(2048, conditions.shape[1])
    game = ConditionalPG09(adapter, rows, critic, conditions=conditions, particles=slider.particles,
        preset=preset, total_steps=3, batch_size=2, response_domain='continuous_vector', **options)
    return backend, adapter, game


def main():
    torch.set_num_threads(1)
    gate = validate_pg09_domain((1, 2048), response_domain='continuous_vector',
        pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    receipts = []
    for preset in ('bispf', 'bispf_optimized'):
        backend, adapter, game = build(preset)
        baseline = _frozen_digest(backend.model)
        bank = adapter.particles.detach().clone()
        first = game.step()
        assert first['d_pen'] > 0
        state = game.state_dict()
        expected_metrics = game.step()
        expected = game.state_dict()
        probes = [game.evaluate(i, seed=19) for i in range(2)]
        assert all(value.shape == (1, 2048) for value in probes)
        assert not torch.equal(bank, adapter.particles)
        assert _frozen_digest(backend.model) == baseline
        other_backend, other, receiver = build(preset)
        receiver.load_state_dict(state)
        equal(receiver.step(), expected_metrics)
        equal(receiver.state_dict(), expected)
        equal([receiver.evaluate(i, seed=19) for i in range(2)], probes)
        exported = receiver.export_state_dict()
        with torch.no_grad():
            next(other.network.parameters()).add_(0.125)
        receiver.load_export_state_dict(exported)
        equal([receiver.evaluate(i, seed=19) for i in range(2)], probes)
        assert _frozen_digest(other_backend.model) == baseline
        assert expected['config']['recipe']['training_precision'] == 'float32'
        assert expected['config']['pg_version'] == '0.9.1'
        receipts.append(dict(preset=preset, response_shape=[1, 2048], d_pen=first['d_pen'],
                             bank_changed=True, frozen_base_unchanged=True, resume_exact=True,
                             export_reload_exact=True, pg_sources_sha256=expected['config']['pg_sources_sha256']))
        del backend, adapter, game, other_backend, other, receiver
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(scope='synthetic width2048 product AR binding; no pretrained runtime/audio',
                         gate=gate, cases=receipts, cuda_initialized=False), indent=2))


if __name__ == '__main__':
    main()
