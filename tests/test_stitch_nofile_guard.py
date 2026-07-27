"""Tests for the `RLIMIT_NOFILE` guard on the slide stitch (DISCOVERED #4).

`_join_overlay_tiles` holds every one of a slide's overlay tiles open as a lazy
pyvips image simultaneously — 27,565 of them at real scale. This host's soft limit
is 1,048,576 so it has always passed here, but the common Linux default is 1,024,
where the stitch would fail *after* the entire multi-hour analysis had completed.

The guard has to do two things: raise the soft limit itself when the hard limit
allows it (the usual case, and no privilege is needed for soft→hard), and fail with
an actionable message when it does not.

The limits are faked rather than really applied: lowering a *hard* limit is
irreversible for the life of a process, so a test that really did it would poison
every test after it.
"""
from __future__ import annotations

import resource

import pytest

import hybrid_pipeline as HP


@pytest.fixture
def fake_limits(monkeypatch):
    """Install a fake RLIMIT_NOFILE; returns a setter and the observed state."""
    state = {"soft": 1024, "hard": 1048576, "set_calls": []}

    def fake_get(which):
        assert which == resource.RLIMIT_NOFILE
        return (state["soft"], state["hard"])

    def fake_set(which, limits):
        assert which == resource.RLIMIT_NOFILE
        soft, hard = limits
        if hard != state["hard"] and state["hard"] != resource.RLIM_INFINITY:
            raise ValueError("not allowed to raise maximum limit")
        if soft > hard != resource.RLIM_INFINITY:
            raise ValueError("current limit exceeds maximum limit")
        state["set_calls"].append((soft, hard))
        state["soft"] = soft

    monkeypatch.setattr(resource, "getrlimit", fake_get)
    monkeypatch.setattr(resource, "setrlimit", fake_set)
    return state


def test_no_action_when_the_soft_limit_already_suffices(fake_limits):
    fake_limits.update(soft=4096, hard=1048576)
    HP._ensure_nofile_limit(1000)
    assert fake_limits["set_calls"] == []


def test_raises_the_soft_limit_when_the_hard_limit_allows(fake_limits):
    """The 1,024-default host: the fix is available without privileges."""
    fake_limits.update(soft=1024, hard=1048576)
    HP._ensure_nofile_limit(30000)
    assert fake_limits["soft"] >= 30000


def test_infinite_hard_limit_is_treated_as_sufficient(fake_limits):
    fake_limits.update(soft=1024, hard=resource.RLIM_INFINITY)
    HP._ensure_nofile_limit(30000)
    assert fake_limits["soft"] >= 30000


def test_infinite_soft_limit_needs_no_change(fake_limits):
    fake_limits.update(soft=resource.RLIM_INFINITY, hard=resource.RLIM_INFINITY)
    HP._ensure_nofile_limit(30000)
    assert fake_limits["set_calls"] == []


def test_fails_loudly_when_the_hard_limit_is_too_low(fake_limits):
    fake_limits.update(soft=512, hard=2048)
    with pytest.raises(RuntimeError) as exc:
        HP._ensure_nofile_limit(30000)
    # the message must tell an operator what to change, and that the analysis
    # output is already on disk — this failure lands after hours of compute
    msg = str(exc.value)
    assert "RLIMIT_NOFILE" in msg
    assert "30000" in msg
    assert "_stitch_scratch/" in msg


def test_stitch_checks_the_limit_before_opening_any_tile(tmp_path, fake_limits):
    """The guard must fire on tile *count*, not on the first EMFILE — otherwise it
    is just a worse error message for the same failure."""
    from m0_stitch import compute_tile_geometry

    stride = 768
    positions = [(x * stride, y * stride) for y in range(4) for x in range(4)]
    geometry = compute_tile_geometry(positions, 1024, 256)

    fake_limits.update(soft=100, hard=200)      # below 16 tiles + 256 headroom
    (tmp_path / "overlay_annotated").mkdir()
    # no tile files exist at all: if the guard runs first we get its RuntimeError;
    # if it does not, we would get FileNotFoundError from the first open
    with pytest.raises(RuntimeError, match="RLIMIT_NOFILE"):
        HP._join_overlay_tiles(tmp_path, geometry)


def test_headroom_is_reserved_above_the_tile_count(fake_limits):
    """The process has its own files open (models, logs, the output TIFF); a limit
    of exactly n_tiles would still EMFILE."""
    fake_limits.update(soft=1024, hard=1048576)
    HP._ensure_nofile_limit(27565 + 256)
    assert fake_limits["soft"] > 27565
