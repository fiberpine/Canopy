#!/usr/bin/env python3
"""Regenerate repository-files.json from the files Git tracks.

Run after adding, changing or deleting tracked files, then check the result
with tools/verify_repository.py. The inventory itself is not listed in it.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "repository-files.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files():
    """Return {path: is_symlink} for every tracked file, in Git's order."""
    output = subprocess.check_output(
        ["git", "-c", "core.quotepath=off", "ls-files", "-s", "-z"], cwd=str(ROOT)
    ).decode("utf-8")
    entries = {}
    for record in output.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        entries[path] = meta.split()[0] == "120000"
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", help="replace the free-text scope field")
    args = parser.parse_args()

    previous = json.loads(INVENTORY.read_text(encoding="utf-8"))
    files = []
    for path, is_symlink in sorted(tracked_files().items()):
        if path == INVENTORY.name:
            continue
        full = ROOT / path
        if is_symlink:
            # On checkouts without symlink support Git stores the target as file text.
            target = os.readlink(str(full)) if full.is_symlink() else full.read_text().strip()
            files.append({"path": path, "symlink": target})
        else:
            files.append({"path": path, "bytes": full.stat().st_size, "sha256": sha256(full)})

    inventory = {
        "scope": args.scope or previous["scope"],
        "original_source_archive_sha256": previous["original_source_archive_sha256"],
        "files": files,
    }
    INVENTORY.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print("wrote %s: %d files" % (INVENTORY.name, len(files)))


if __name__ == "__main__":
    main()
