import pytest

from spamguard.registry import ModelRegistry


def _artifact(tmp_path, name="a"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "model.bin").write_text("x")
    return directory


def test_register_versions_increment(tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    v1 = registry.register("svm", _artifact(tmp_path, "a"), {"macro_f1": 0.8}, {})
    v2 = registry.register("bigru", _artifact(tmp_path, "b"), {"macro_f1": 0.9}, {})
    assert (v1, v2) == (1, 2)
    assert registry.get_meta(2)["model_type"] == "bigru"


def test_promote_sets_alias_and_history(tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    version = registry.register("svm", _artifact(tmp_path), {"macro_f1": 0.8}, {})
    registry.promote(version, reason="best")
    assert registry.get_alias("production") == version
    assert registry.list_versions()["history"][-1]["reason"] == "best"


def test_promote_unknown_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path / "reg")
    with pytest.raises(KeyError):
        registry.promote(99)
