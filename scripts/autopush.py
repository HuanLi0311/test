#!/usr/bin/python3
"""Commit and push this workspace, keeping files >= 50 MiB local."""
import datetime
import fcntl
import os
from pathlib import Path
import stat
import subprocess

LIMIT = 50 * 1024 * 1024


def git(root, *args, **kwargs):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        stdout=subprocess.PIPE, timeout=120, **kwargs
    ).stdout


def ignore_pattern(path):
    if "\n" in path or "\r" in path:
        raise ValueError("Rename large file containing a newline before syncing")
    return "/" + "".join("\\" + c if c in "\\*?[] \t" else c for c in path)


def sync(root):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root.resolve():
        raise RuntimeError("Refusing to operate on a parent repository")
    git_dir = Path(os.fsdecode(git(root, "rev-parse", "--absolute-git-dir")).strip())
    with (git_dir / "autopush.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if git(root, "symbolic-ref", "--short", "HEAD").strip() != b"main":
            raise RuntimeError("Automatic sync requires branch main")
        for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "BISECT_START"):
            if (git_dir / marker).exists():
                raise RuntimeError("Finish the in-progress Git operation before syncing")

        large = []
        paths = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        for raw in sorted(set(paths.split(b"\0")) - {b""}):
            path = os.fsdecode(raw)
            try:
                info = (root / path).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_size >= LIMIT:
                large.append(ignore_pattern(path))
                print("Keeping large file local:", repr(path), flush=True)
        if large:
            if (root / ".gitignore").is_symlink():
                raise RuntimeError(".gitignore must be a regular file")
            with (root / ".gitignore").open("a", encoding="utf-8", errors="surrogateescape") as out:
                out.write("\n" + "\n".join(large) + "\n")

        ignored = git(root, "ls-files", "--cached", "--ignored", "--exclude-standard", "-z")
        if ignored:
            git(root, "update-index", "--force-remove", "-z", "--stdin", input=ignored)
        git(root, "add", "--all")

        # ponytail: scan the whole index; use incremental checks only if this grows slow.
        entries = git(root, "ls-files", "--stage", "-z").split(b"\0")
        objects = set()
        for entry in filter(None, entries):
            metadata, path = entry.split(b"\t", 1)
            mode, oid, stage = metadata.split()
            if mode == b"160000" or stage != b"0":
                raise RuntimeError("Resolve nested repositories or unmerged files before syncing")
            objects.add(oid)
        sizes = git(root, "cat-file", "--batch-check=%(objectsize)", input=b"\n".join(objects) + b"\n") if objects else b""
        if any(int(size) >= LIMIT for size in sizes.splitlines()):
            raise RuntimeError("A staged file grew to >= 50 MiB; retry after the writer finishes")

        changed = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet"]).returncode
        if changed == 1:
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            print(os.fsdecode(git(root, "commit", "-m", "Auto-sync RNA workspace " + stamp)), flush=True)
        elif changed:
            raise RuntimeError("Could not inspect staged changes")
        # A normal push preserves remote history; divergence is logged for manual resolution.
        print(os.fsdecode(git(root, "push", "--porcelain", "origin", "HEAD:refs/heads/main")), flush=True)


if __name__ == "__main__":
    sync(Path(__file__).resolve().parents[1])
