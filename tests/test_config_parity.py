"""Guard `config.py` / `config_example.py` parity (doc 07 G2, DISCOVERED #36).

`config.py` is gitignored — every checkout starts by copying `config_example.py`
over it. That makes the example file the *only* checked-in description of the
config surface, and lets the two drift silently: G2 was exactly that, the example
file having lost the `compute_config_hash()` / `config = Config()` tail, and it was
caught by someone reading the file rather than by any check.

Parity here means the **config surface**, not byte equality: field names, types,
declaration order and default values, plus the module tail every caller imports.
Comments and site-local *values* inside `config.py` are deliberately out of scope —
that file is where a deployment writes its own paths.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
LIVE = HYBRID / "config.py"
EXAMPLE = HYBRID / "config_example.py"

# Fields whose default is deployment-specific: the example ships a placeholder and
# a real config.py is expected to point somewhere else. Everything else must match.
SITE_LOCAL = {
    "base_dir", "unet_model_path", "cellpose_model_path", "cellpose_dish_model_path",
    "ihc_tile_dir", "dish_tile_dir", "output_dir",
    "ihc_test_path", "dish_test_path", "device",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def modules():
    if not LIVE.exists():
        pytest.skip("config.py absent — copy config_example.py to create it")
    return _load(LIVE, "_cfg_live"), _load(EXAMPLE, "_cfg_example")


def test_both_expose_the_module_tail_callers_import(modules):
    """G2's actual bug: the example file had lost these two."""
    for mod in modules:
        assert hasattr(mod, "Config"), f"{mod.__name__} has no Config"
        assert hasattr(mod, "compute_config_hash"), f"{mod.__name__} has no hash fn"
        assert hasattr(mod, "config"), f"{mod.__name__} has no `config` singleton"
        assert isinstance(mod.config, mod.Config)


def test_field_names_and_order_match(modules):
    live, example = modules
    assert [f.name for f in dataclasses.fields(live.Config)] == \
           [f.name for f in dataclasses.fields(example.Config)]


def test_field_types_match(modules):
    live, example = modules
    lf = {f.name: f.type for f in dataclasses.fields(live.Config)}
    ef = {f.name: f.type for f in dataclasses.fields(example.Config)}
    assert lf == ef


def test_non_site_local_defaults_match(modules):
    """A tuning parameter that differs between the two is drift, not configuration."""
    live, example = modules
    lf = {f.name: f for f in dataclasses.fields(live.Config)}
    ef = {f.name: f for f in dataclasses.fields(example.Config)}
    mismatched = {}
    for name in lf:
        if name in SITE_LOCAL:
            continue
        a, b = getattr(live.config, name), getattr(example.config, name)
        if a != b:
            mismatched[name] = (a, b)
    assert not mismatched, f"defaults drifted (live, example): {mismatched}"


def test_site_local_allowlist_only_names_real_fields(modules):
    """Stops the allowlist from quietly hiding a field that has since been renamed."""
    live, _ = modules
    names = {f.name for f in dataclasses.fields(live.Config)}
    assert SITE_LOCAL <= names, f"stale allowlist entries: {SITE_LOCAL - names}"


def test_config_hash_is_stable_and_field_sensitive(modules):
    live, _ = modules
    assert live.compute_config_hash(live.config) == live.compute_config_hash(live.config)
    other = dataclasses.replace(live.config, default_tile_size=512)
    assert live.compute_config_hash(other) != live.compute_config_hash(live.config)


def test_runtime_only_knobs_are_excluded_from_the_hash(modules):
    """Fields that change *how* a run executes but not *what* it produces must not
    enter the hash.

    Two things break if they do. Round-over-round comparison stops working even though
    no algorithm changed — the hash is written into every CSV precisely so results can
    be matched across runs. And, worse, `_mp_tile_worker` compares the parent's hash to
    its own: a `spawn`ed worker re-imports `config` fresh and cannot see a runtime
    mutation of the parent's singleton, so the whole batch fail-fasts. That is not
    hypothetical — `scripts/alloc_conf_probe.py` hit it on its first run.
    """
    for mod in modules:
        base = mod.compute_config_hash(mod.config)
        for field in mod._HASH_EXCLUDE:
            assert hasattr(mod.config, field), f"stale _HASH_EXCLUDE entry: {field}"
            current = getattr(mod.config, field)
            changed = f"{current}-changed" if isinstance(current, str) else not current
            mutated = dataclasses.replace(mod.config, **{field: changed})
            assert mod.compute_config_hash(mutated) == base, \
                f"{mod.__name__}: {field} leaks into the config hash"


def test_hash_exclusions_match_between_the_two_files(modules):
    live, example = modules
    assert live._HASH_EXCLUDE == example._HASH_EXCLUDE
