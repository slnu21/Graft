import pytest

from anograft.core.scale import physical_area_um2, physical_scale


def test_factor_is_source_over_target() -> None:
    r = physical_scale(5.5, 2.75)
    assert r.applied and r.reason is None
    assert r.factor == pytest.approx(2.0)


def test_missing_either_side_is_identity_with_reason() -> None:
    for s, t in [(None, 2.75), (5.5, None), (None, None)]:
        r = physical_scale(s, t)
        assert r.factor == 1.0
        assert r.applied is False
        assert r.reason


def test_non_positive_pitch_rejected() -> None:
    with pytest.raises(ValueError):
        physical_scale(0.0, 1.0)
    with pytest.raises(ValueError):
        physical_scale(1.0, -1.0)


def test_physical_area() -> None:
    assert physical_area_um2(100, 2.0) == pytest.approx(400.0)
    assert physical_area_um2(100, None) is None
