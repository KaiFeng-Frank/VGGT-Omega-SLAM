#!/usr/bin/env python3
"""Fetch fixed upstream Git revisions and verify their tracked files.

This installs source checkouts only. It does not install Python packages,
build GTSAM, download checkpoints, or use a GPU. Existing checkouts must
already match the pinned revision; they are never reset or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vos import env


def git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to fetch and verify upstream sources.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Git failed in {path}: {exc.stderr.strip()}") from exc


def verify_checkout(path: Path, spec: dict) -> dict:
    if not (path / ".git").exists():
        raise RuntimeError(f"{path} is not a Git checkout. Use a fresh destination; existing files will not be overwritten.")
    actual = git(path, "rev-parse", "HEAD").strip()
    if actual != spec["commit"]:
        raise RuntimeError(f"{path} is at {actual}; expected {spec['commit']}. Use a fresh directory or correct it manually.")
    files = []
    for record in git(path, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if not record:
            continue
        metadata, relative = record.split("\t", 1)
        mode, object_type, expected = metadata.split()
        if object_type != "blob":
            raise RuntimeError(f"Unsupported Git object {object_type}: {relative}")
        file_path = path / relative
        if mode == "120000":
            if not file_path.is_symlink():
                raise RuntimeError(f"Expected source symlink: {file_path}")
            data = os.readlink(file_path).encode()
        elif file_path.is_file():
            data = file_path.read_bytes()
        else:
            raise RuntimeError(f"Missing upstream file: {file_path}")
        actual_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual_blob != expected:
            raise RuntimeError(f"Modified upstream file: {file_path}")
        files.append({"path": relative, "git_blob": expected, "sha256": hashlib.sha256(data).hexdigest()})
    return {"repository": spec["url"], "commit": actual, "files": files, "tracked_files_verified": len(files)}


def fetch_source(path: Path, spec: dict, verify_only: bool = False) -> dict:
    if not path.exists():
        if verify_only:
            raise RuntimeError(f"Missing source checkout: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        git(path.parent, "init", "--quiet", str(path))
        git(path, "remote", "add", "origin", spec["url"])
        git(path, "fetch", "--depth", "1", "origin", spec["commit"])
        git(path, "checkout", "--detach", "FETCH_HEAD")
    manifest = verify_checkout(path, spec)
    if not verify_only:
        (path / ".vos-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=list(env.SOURCES), default=list(env.SOURCES))
    parser.add_argument("--deps", type=Path, help="override destination base; otherwise respect VOS_DEPS/VOS_*_SRC")
    parser.add_argument("--verify-only", action="store_true", help="read existing checkouts without network or writes")
    args = parser.parse_args(argv)
    try:
        for name in args.sources:
            spec = env.SOURCES[name]
            path = (args.deps.expanduser().resolve() / spec["path"].name) if args.deps else spec["path"]
            print(f"[{name}] {path} @ {spec['commit']}", flush=True)
            manifest = fetch_source(path, spec, args.verify_only)
            if name == "slam":
                expected = json.loads(env.SLAM_PROVENANCE.read_text())
                actual = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
                if expected["commit"] != manifest["commit"] or any(
                    actual.get(entry["path"]) != entry["sha256"] for entry in expected["files"]
                ):
                    raise RuntimeError("Fetched SLAM tree does not match the bundled provenance manifest")
            print(f"[{name}] verified {manifest['tracked_files_verified']} tracked files", flush=True)
    except (RuntimeError, OSError) as exc:
        print(f"Source setup failed: {exc}", file=sys.stderr)
        return 1
    print("Sources verified. Install runtime dependencies and GTSAM with SL(4) separately; download approved checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
