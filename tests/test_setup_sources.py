"""Check source provenance verification with a tiny mocked Git tree; no downloads."""
import hashlib

import pytest

from tools import setup_sources as setup


def test_source_verification_detects_tampering_and_verify_only_does_not_write(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    data = b"original source\n"
    path = source / "module.py"
    path.write_bytes(data)
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    spec = {"url": "https://example.invalid/source.git", "commit": "1" * 40}
    def fake_git(_path, *args):
        if args == ("rev-parse", "HEAD"):
            return spec["commit"] + "\n"
        if args == ("ls-tree", "-r", "-z", "HEAD"):
            return f"100644 blob {blob}\tmodule.py\0"
        pytest.fail(f"verify-only attempted Git mutation: {args}")
    monkeypatch.setattr(setup, "git", fake_git)
    result = setup.fetch_source(source, spec, verify_only=True)
    assert result["tracked_files_verified"] == 1
    assert not (source / ".vos-source.json").exists()
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Modified upstream file"):
        setup.verify_checkout(source, spec)


def test_existing_non_git_files_are_never_overwritten(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(RuntimeError, match="not a Git checkout"):
        setup.fetch_source(existing, {"url": "unused", "commit": "1" * 40})
    assert marker.read_text() == "keep"
