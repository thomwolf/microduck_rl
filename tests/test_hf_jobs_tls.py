from pathlib import Path

from mjlab_microduck.hf_jobs import _first_existing_ca_bundle


def test_ca_bundle_prefers_existing_python_default(tmp_path: Path):
    configured = tmp_path / "configured.pem"
    fallback = tmp_path / "fallback.pem"
    configured.touch()
    fallback.touch()

    selected = _first_existing_ca_bundle(str(configured), (fallback,))

    assert selected == configured


def test_ca_bundle_falls_back_when_python_default_is_missing(tmp_path: Path):
    missing = tmp_path / "missing.pem"
    fallback = tmp_path / "fallback.pem"
    fallback.touch()

    selected = _first_existing_ca_bundle(str(missing), (fallback,))

    assert selected == fallback
