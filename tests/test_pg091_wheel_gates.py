"""Installed-wheel and old-lineage rejection without shared source checkouts."""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch


class DistributionView:
    """Read installed metadata while optionally relocating copied package bytes."""

    def __init__(self, distribution, *, root=None, direct_url=None, version=None):
        self.distribution = distribution
        self.root = root
        self.direct_url = direct_url
        self.version = distribution.version if version is None else version

    def locate_file(self, path):
        if self.root is not None:
            return self.root / path
        return self.distribution.locate_file(path)

    def read_text(self, name):
        if name == "direct_url.json" and self.direct_url is not None:
            return json.dumps(self.direct_url)
        return self.distribution.read_text(name)


def runtime_lock():
    from yue2.integration import LOCK_PATH

    return json.loads(LOCK_PATH.read_text())


@pytest.fixture(autouse=True)
def source_free_cpu(monkeypatch):
    for name in ("PYTHONPATH", "YUE2_AUDIO_SOURCE", "YUE2_CORE_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_tampered_copied_install_fails_before_shared_import(tmp_path, monkeypatch):
    from yue2 import integration

    integration.configure_integration()
    original_distribution = integration._distribution
    record = runtime_lock()["core"]
    installed = metadata.distribution(record["name"])
    original = Path(installed.locate_file("particle_sliders/pg09_audio.py"))
    original_bytes = original.read_bytes()
    copied_root = tmp_path / "copied-install"
    shutil.copytree(installed.locate_file("particle_sliders"), copied_root / "particle_sliders")
    copied = copied_root / "particle_sliders/pg09_audio.py"
    copied.write_bytes(original_bytes + b"\n# temporary tamper regression\n")
    view = DistributionView(installed, root=copied_root)

    def copied_distribution(candidate):
        return view if candidate["name"] == record["name"] else original_distribution(candidate)

    monkeypatch.setattr(integration, "_distribution", copied_distribution)
    original_path = list(sys.path)
    with pytest.raises(ValueError, match="sources differ from the pinned wheel"):
        integration.configure_integration()
    assert sys.path == original_path
    assert original.read_bytes() == original_bytes
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("family", ("core", "particlegan"))
def test_distribution_accepts_the_exact_locked_wheel_hash(family, monkeypatch):
    from yue2 import integration

    record = runtime_lock()[family]
    installed = metadata.distribution(record["name"])
    # Reuse the actual installed direct_url receipt, including its wheel hash.
    view = DistributionView(installed)
    monkeypatch.setattr(integration.metadata, "distribution", lambda _name: view)
    assert integration._distribution(record) is view


@pytest.mark.parametrize("family", ("core", "particlegan"))
@pytest.mark.parametrize("direct_url", (
    {"url": "https://example.invalid/foreign.whl", "archive_info": {"hashes": {"sha256": "0" * 64}}},
    {"url": "https://example.invalid/foreign.git", "vcs_info": {"vcs": "git", "commit_id": "0" * 40}},
    {},
))
def test_distribution_rejects_foreign_or_unbound_wheel_identity(family, direct_url, monkeypatch):
    from yue2 import integration

    record = runtime_lock()[family]
    installed = metadata.distribution(record["name"])
    view = DistributionView(installed, direct_url=direct_url)
    monkeypatch.setattr(integration.metadata, "distribution", lambda _name: view)
    with pytest.raises(ValueError, match="wheel identity differs"):
        integration._distribution(record)


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            equal_tree(first, second)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def test_old_pg090_full_checkpoint_is_rejected_without_mutation(tmp_path, monkeypatch):
    from particle_sliders import ConditionalPG09
    from yue2.integration import configure_integration
    from yue2.train import parse_args, train

    configure_integration()
    first, rejected = tmp_path / "first", tmp_path / "rejected"

    def arguments(destination, *extra):
        return ["--dummy", "--name", "lineage", "--save_dir", str(destination),
                "--steps", "3", "--save_every", "1", "--seed", "7", *extra]

    train(parse_args(arguments(first, "--until", "1")))
    checkpoint = first / "lineage_checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    packet = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert packet["game"]["config"]["pg_version"] == "0.9.1"
    assert packet["game"]["completed_steps"] == 1
    packet["game"]["config"]["pg_version"] = "0.9.0"
    historical = tmp_path / "historical-pg090.pt"
    torch.save(packet, historical)
    original_historical = historical.read_bytes()
    original_restore = ConditionalPG09.load_state_dict
    rejected_receivers = []

    def observe_rejection(game, state):
        before = deepcopy(game.state_dict())
        try:
            return original_restore(game, state)
        except ValueError:
            equal_tree(game.state_dict(), before)
            rejected_receivers.append(game)
            raise

    monkeypatch.setattr(ConditionalPG09, "load_state_dict", observe_rejection)
    with pytest.raises(ValueError, match="mismatch"):
        train(parse_args(arguments(rejected, "--resume", str(historical))))
    assert len(rejected_receivers) == 1
    assert not rejected.exists()
    assert checkpoint.read_bytes() == original_checkpoint
    assert historical.read_bytes() == original_historical
    assert not torch.cuda.is_initialized()


def install_sparse_clock_critic(monkeypatch):
    from particle_sliders import pg09_audio

    original_critic = pg09_audio.AudioResponseCritic

    class CriticWithSparseParameter(original_critic):
        def __init__(self, width, condition_width):
            super().__init__(width, condition_width)
            self.optional = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("optional_active", torch.tensor(False))

        def forward(self, response, condition):
            value = super().forward(response, condition)
            if self.optional_active:
                value = value + (response[:, :2] * self.optional).sum(dim=1)
            return value

    monkeypatch.setattr(pg09_audio, "AudioResponseCritic", CriticWithSparseParameter)


def optimizer_identifier(packet, optimizer, parameter):
    return next(key for saved, actual in zip(packet["param_groups"], optimizer.param_groups)
                for key, owned in zip(saved["params"], actual["params"]) if owned is parameter)


def assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, *, preset, optimizer_name, mutation, initial_steps=1, sparse_critic=False):
    from particle_sliders import ConditionalPG09
    from yue2.integration import configure_integration
    from yue2.train import parse_args, train

    configure_integration()
    if sparse_critic:
        install_sparse_clock_critic(monkeypatch)
    first, rejected = tmp_path / "first", tmp_path / "rejected"

    def arguments(destination, *extra):
        return ["--dummy", "--name", "optimizer-gate", "--save_dir", str(destination),
                "--steps", "3", "--save_every", "1", "--seed", "7", "--preset", preset, *extra]

    original_step = ConditionalPG09.step
    initial_games = []

    def observe_initial_game(game):
        metrics = original_step(game)
        if sparse_critic and game.completed_steps == 1:
            game.critic.optional_active.fill_(True)
        initial_games.append(game)
        return metrics

    with monkeypatch.context() as initial_observer:
        initial_observer.setattr(ConditionalPG09, "step", observe_initial_game)
        train(parse_args(arguments(first, "--until", str(initial_steps))))
    assert len(initial_games) == initial_steps
    checkpoint = first / "optimizer-gate_checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    packet = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert packet["game"]["completed_steps"] == initial_steps
    histories = packet["game"][optimizer_name]["state"]
    changed = 0
    if mutation in ("clock-zero", "clock-low", "clock-high"):
        if sparse_critic:
            source = initial_games[-1]
            identifier = optimizer_identifier(
                packet["game"]["opt_d"], source.opt_d, source.critic.optional,
            )
            assert histories[identifier]["step"].item() == 1
        else:
            identifier = next(key for key, history in histories.items() if history)
            assert histories[identifier]["step"].item() == initial_steps
        count = {"clock-zero": 0, "clock-low": initial_steps - 1, "clock-high": initial_steps + 1}[mutation]
        histories[identifier]["step"].fill_(count)
        changed = 1
    elif mutation in ("deleted-history", "emptied-history", "extra-history-field"):
        identifier = next(key for key, history in histories.items() if history)
        if mutation == "deleted-history":
            del histories[identifier]
        elif mutation == "emptied-history":
            histories[identifier] = {}
        else:
            histories[identifier]["unexpected_history_field"] = 1
        changed = 1
    else:
        for history in histories.values():
            if "exp_avg_sq" in history:
                if mutation == "negative-second-moment":
                    history["exp_avg_sq"].fill_(-1.0)
                elif mutation == "missing-second-moment":
                    del history["exp_avg_sq"]
                else:
                    raise AssertionError("Unknown test optimizer mutation")
                changed += 1
    if not changed and mutation == "negative-second-moment":
        # The optimized preset uses DualNorm/RowNorm rather than Adam history.
        # A supplied Adam slot must still reject a negative second moment; use
        # the actual owned leaf schema so this cannot pass via a shape error.
        optimizer = getattr(initial_games[0], optimizer_name)
        parameter = optimizer.param_groups[0]["params"][0]
        identifier = packet["game"][optimizer_name]["param_groups"][0]["params"][0]
        histories[identifier]["exp_avg_sq"] = torch.full_like(parameter.detach().cpu(), -1.0)
        changed = 1
    assert changed, f"{preset} {optimizer_name} did not create Adam second-moment history"
    incompatible = tmp_path / f"{optimizer_name}-{mutation}.pt"
    torch.save(packet, incompatible)
    original_incompatible = incompatible.read_bytes()
    original_restore = ConditionalPG09.load_state_dict
    rejected_receivers = []

    def observe_rejection(game, state):
        before = deepcopy(game.state_dict())
        try:
            original_restore(game, state)
        except ValueError:
            equal_tree(game.state_dict(), before)
            rejected_receivers.append(game)
            raise
        else:
            pytest.fail(f"Resume accepted invalid {preset} {optimizer_name} history: {mutation}")

    monkeypatch.setattr(ConditionalPG09, "load_state_dict", observe_rejection)
    with pytest.raises(ValueError, match="(?i)optimizer|adam|moment|history"):
        train(parse_args(arguments(rejected, "--resume", str(incompatible))))
    assert len(rejected_receivers) == 1
    assert not rejected.exists()
    assert checkpoint.read_bytes() == original_checkpoint
    assert incompatible.read_bytes() == original_incompatible
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("preset", ("bispf", "bispf_optimized"))
@pytest.mark.parametrize("optimizer_name", ("opt_g", "opt_d"))
def test_resume_negative_adam_second_moment_rejects_before_mutation_and_output(
        tmp_path, monkeypatch, preset, optimizer_name):
    assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, preset=preset, optimizer_name=optimizer_name,
        mutation="negative-second-moment",
    )


@pytest.mark.parametrize("optimizer_name", ("opt_g", "opt_d"))
def test_resume_bispf_missing_adam_history_rejects_before_mutation_and_output(
        tmp_path, monkeypatch, optimizer_name):
    assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, preset="bispf", optimizer_name=optimizer_name,
        mutation="missing-second-moment",
    )


@pytest.mark.parametrize("preset", ("bispf", "bispf_optimized"))
@pytest.mark.parametrize("optimizer_name", ("opt_g", "opt_d"))
@pytest.mark.parametrize("mutation", ("deleted-history", "emptied-history", "extra-history-field"))
def test_resume_started_optimizer_history_rejects_before_mutation_and_output(
        tmp_path, monkeypatch, preset, optimizer_name, mutation):
    assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, preset=preset, optimizer_name=optimizer_name, mutation=mutation,
    )


@pytest.mark.parametrize("preset", ("bispf", "bispf_optimized"))
def test_initial_empty_optimizer_history_remains_valid_product_resume(tmp_path, monkeypatch, preset):
    from particle_sliders import ConditionalPG09
    from yue2.integration import configure_integration
    from yue2.train import parse_args, train

    configure_integration()
    first, resumed, uninterrupted = [tmp_path / name for name in ("first", "resumed", "uninterrupted")]

    def arguments(destination, *extra):
        return ["--dummy", "--name", "initial-history", "--save_dir", str(destination),
                "--steps", "3", "--save_every", "1", "--seed", "7", "--preset", preset, *extra]

    original_step = ConditionalPG09.step
    initial_states = []

    def capture_initial_state(game):
        initial_states.append(game.state_dict())
        return original_step(game)

    with monkeypatch.context() as capture:
        capture.setattr(ConditionalPG09, "step", capture_initial_state)
        train(parse_args(arguments(first, "--until", "1")))
    assert len(initial_states) == 1
    initial = initial_states[0]
    assert initial["completed_steps"] == 0
    assert initial["opt_g"]["state"] == initial["opt_d"]["state"] == {}
    packet = torch.load(first / "initial-history_checkpoint.pt", map_location="cpu", weights_only=True)
    packet.update(game=initial, metrics=[], probes=[])
    initial_checkpoint = tmp_path / "initial.pt"
    torch.save(packet, initial_checkpoint)
    original_checkpoint = initial_checkpoint.read_bytes()
    original_restore = ConditionalPG09.load_state_dict
    restored = []

    def observe_initial_restore(game, state):
        original_restore(game, state)
        equal_tree(game.state_dict(), initial)
        restored.append(game)

    with monkeypatch.context() as observe:
        observe.setattr(ConditionalPG09, "load_state_dict", observe_initial_restore)
        train(parse_args(arguments(resumed, "--resume", str(initial_checkpoint))))
    assert len(restored) == 1
    train(parse_args(arguments(uninterrupted)))
    actual = torch.load(resumed / "initial-history_checkpoint.pt", map_location="cpu", weights_only=True)
    expected = torch.load(uninterrupted / "initial-history_checkpoint.pt", map_location="cpu", weights_only=True)
    equal_tree(actual, expected)
    assert initial_checkpoint.read_bytes() == original_checkpoint
    assert actual["game"]["completed_steps"] == 3
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("preset", ("bispf", "bispf_optimized"))
def test_honest_unused_critic_history_remains_valid_product_resume(tmp_path, monkeypatch, preset):
    from particle_sliders import ConditionalPG09
    from particle_sliders import pg09_audio
    from yue2.integration import configure_integration
    from yue2.train import parse_args, train

    configure_integration()
    original_critic = pg09_audio.AudioResponseCritic

    class CriticWithUnusedParameter(original_critic):
        def __init__(self, width, condition_width):
            super().__init__(width, condition_width)
            self.unused = torch.nn.Parameter(torch.ones(2))

    monkeypatch.setattr(pg09_audio, "AudioResponseCritic", CriticWithUnusedParameter)
    first, resumed, uninterrupted = [tmp_path / name for name in ("first", "resumed", "uninterrupted")]

    def arguments(destination, *extra):
        return ["--dummy", "--name", "unused-critic", "--save_dir", str(destination),
                "--steps", "3", "--save_every", "1", "--seed", "7", "--preset", preset, *extra]

    original_step = ConditionalPG09.step
    first_games = []

    def observe_first_game(game):
        result = original_step(game)
        first_games.append(game)
        return result

    with monkeypatch.context() as observe:
        observe.setattr(ConditionalPG09, "step", observe_first_game)
        train(parse_args(arguments(first, "--until", "1")))
    assert len(first_games) == 1
    source = first_games[0]
    checkpoint = first / "unused-critic_checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    packet = torch.load(checkpoint, map_location="cpu", weights_only=True)
    unused = next(key for saved, actual in zip(packet["game"]["opt_d"]["param_groups"], source.opt_d.param_groups)
                  for key, parameter in zip(saved["params"], actual["params"]) if parameter is source.critic.unused)
    assert unused not in packet["game"]["opt_d"]["state"]
    assert packet["game"]["optimizer_history_schema"]["d"][unused] is None
    train(parse_args(arguments(resumed, "--resume", str(checkpoint))))
    train(parse_args(arguments(uninterrupted)))
    actual = torch.load(resumed / "unused-critic_checkpoint.pt", map_location="cpu", weights_only=True)
    expected = torch.load(uninterrupted / "unused-critic_checkpoint.pt", map_location="cpu", weights_only=True)
    equal_tree(actual, expected)
    assert unused not in actual["game"]["opt_d"]["state"]
    assert actual["game"]["optimizer_history_schema"]["d"][unused] is None
    assert checkpoint.read_bytes() == original_checkpoint
    assert actual["game"]["completed_steps"] == 3
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ("clock-zero", "clock-low", "clock-high"))
def test_resume_adam_generator_clock_matches_completed_steps_atomically(tmp_path, monkeypatch, mutation):
    assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, preset="bispf", optimizer_name="opt_g", mutation=mutation, initial_steps=2,
    )


@pytest.mark.parametrize("mutation", ("clock-zero", "clock-high"))
def test_resume_sparse_adam_critic_clock_bounds_reject_atomically(tmp_path, monkeypatch, mutation):
    assert_invalid_optimizer_resume_is_transactional(
        tmp_path, monkeypatch, preset="bispf", optimizer_name="opt_d", mutation=mutation,
        initial_steps=2, sparse_critic=True,
    )


def test_sparse_adam_critic_clock_one_at_two_completed_steps_resumes_exactly(tmp_path, monkeypatch):
    from particle_sliders import ConditionalPG09
    from yue2.integration import configure_integration
    from yue2.train import parse_args, train

    configure_integration()
    install_sparse_clock_critic(monkeypatch)
    first, resumed, uninterrupted = [tmp_path / name for name in ("first", "resumed", "uninterrupted")]

    def arguments(destination, *extra):
        return ["--dummy", "--name", "sparse-clock", "--save_dir", str(destination),
                "--steps", "3", "--save_every", "1", "--seed", "7", "--preset", "bispf", *extra]

    original_step = ConditionalPG09.step
    observed = []

    def activate_after_first_update(game):
        metrics = original_step(game)
        if game.completed_steps == 1:
            game.critic.optional_active.fill_(True)
        observed.append(game)
        return metrics

    monkeypatch.setattr(ConditionalPG09, "step", activate_after_first_update)
    train(parse_args(arguments(first, "--until", "2")))
    source = observed[-1]
    checkpoint = first / "sparse-clock_checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    packet = torch.load(checkpoint, map_location="cpu", weights_only=True)
    optional = optimizer_identifier(packet["game"]["opt_d"], source.opt_d, source.critic.optional)
    assert packet["game"]["completed_steps"] == 2
    assert packet["game"]["opt_d"]["state"][optional]["step"].item() == 1
    train(parse_args(arguments(resumed, "--resume", str(checkpoint))))
    train(parse_args(arguments(uninterrupted)))
    actual = torch.load(resumed / "sparse-clock_checkpoint.pt", map_location="cpu", weights_only=True)
    expected = torch.load(uninterrupted / "sparse-clock_checkpoint.pt", map_location="cpu", weights_only=True)
    equal_tree(actual, expected)
    assert checkpoint.read_bytes() == original_checkpoint
    assert actual["game"]["completed_steps"] == 3
    assert actual["game"]["opt_d"]["state"][optional]["step"].item() == 2
    assert not torch.cuda.is_initialized()
