import numpy as np
import pytest

from anograft.core.seeds import bank_fingerprint, image_rng, pipeline_hash, split_rng


def _draw(rng: np.random.Generator, n: int = 8) -> list[float]:
    return rng.random(n).tolist()


def test_same_seed_index_same_stream() -> None:
    assert _draw(image_rng(7, 3)) == _draw(image_rng(7, 3))


def test_order_of_construction_does_not_matter() -> None:
    """워커가 어떤 순서로 이미지를 처리해도 인덱스 i의 스트림은 같다."""
    a_first = _draw(image_rng(7, 0))
    _ = _draw(image_rng(7, 1))
    a_again = _draw(image_rng(7, 0))
    assert a_first == a_again


def test_different_index_different_stream() -> None:
    assert _draw(image_rng(7, 0)) != _draw(image_rng(7, 1))


def test_different_seed_different_stream() -> None:
    assert _draw(image_rng(7, 0)) != _draw(image_rng(8, 0))


def test_split_and_image_purposes_do_not_collide() -> None:
    assert _draw(split_rng(7)) != _draw(image_rng(7, 0))
    assert _draw(split_rng(7)) == _draw(split_rng(7))


def test_negative_index_rejected() -> None:
    with pytest.raises(ValueError):
        image_rng(7, -1)


def test_bank_fingerprint_is_order_independent() -> None:
    a = bank_fingerprint([("scratch/001", 120), ("dent/002", 40)])
    b = bank_fingerprint([("dent/002", 40), ("scratch/001", 120)])
    assert a == b
    assert len(a) == 16
    assert a != bank_fingerprint([("scratch/001", 121), ("dent/002", 40)])


def test_pipeline_hash_changes_with_each_input() -> None:
    base = pipeline_hash("a: 1\n", "0.1.0", "fp")
    assert len(base) == 16
    assert base == pipeline_hash("a: 1\n", "0.1.0", "fp")
    assert base != pipeline_hash("a: 2\n", "0.1.0", "fp")
    assert base != pipeline_hash("a: 1\n", "0.2.0", "fp")
    assert base != pipeline_hash("a: 1\n", "0.1.0", "fq")
