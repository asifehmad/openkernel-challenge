"""Find a supplied CLI task bundle without assuming okc's directory layout."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

TASK_FILES = ("TASK.md", "AGENTS.md", "baseline.py", "reference.py", "checks.py", "submission.py")
CONTRACT_FILES = TASK_FILES[:-1]


def resolve_bundle(path):
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Task bundle path must be a directory")

    def complete(directory):
        return all((directory / name).is_file() for name in TASK_FILES)

    def safe(directory):
        for name in TASK_FILES:
            if not (directory / name).resolve().is_relative_to(root):
                raise ValueError(f"Task file escapes supplied directory: {name}")
        return directory

    if complete(root):
        return safe(root)
    candidates = []

    def onerror(error):
        raise error

    for directory, dirs, _ in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        directory = Path(directory)
        if complete(directory):
            candidates.append(safe(directory))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one complete task bundle under {root}; found {len(candidates)}")
    return candidates[0]


def hashes(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in TASK_FILES}


def require_matching_contract(directory, snapshot):
    actual, expected = hashes(directory), hashes(snapshot)
    changed = [name for name in CONTRACT_FILES if actual[name] != expected[name]]
    if changed:
        raise ValueError("Task contract differs in " + ", ".join(changed) +
                         "; reconcile the new official contract/scoring before running the session")
    return actual
