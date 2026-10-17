#!/usr/bin/python3
"""Run with: /usr/bin/python3 scripts/test_autopush.py (no network needed)."""
import fcntl
from pathlib import Path
import subprocess
import tempfile

from autopush import LIMIT, git, sync


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "work"
    root.mkdir()
    remote = Path(tmp) / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    git(root, "init", "--quiet")
    git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    git(root, "config", "user.name", "Autopush check")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "remote", "add", "origin", str(remote))
    (root / ".gitignore").write_text((Path(__file__).resolve().parents[1] / ".gitignore").read_text())
    (root / "data/reports").mkdir(parents=True)
    (root / "data/DATASETS.md").write_text("dataset documentation\n")
    (root / "data/reports/small.json").write_text("{}\n")
    (root / "data/raw-unknown").write_text("local dataset\n")
    (root / "code.py").write_text("print('ok')\n")
    (root / "tracked").write_text("small\n")
    sync(root)
    first = git(root, "rev-parse", "HEAD")
    sync(root)
    assert git(root, "rev-parse", "HEAD") == first, "Unchanged files made a commit"

    names = ["unknown-format", "odd [data]*? ", "back\\slash", "tracked"]
    for name in names:
        with (root / name).open("wb") as out:
            out.truncate(LIMIT)
    (root / "below-limit").write_bytes(b"small")
    with (root / "below-limit").open("r+b") as out:
        out.truncate(LIMIT - 1)
    (root / "link-to-large").symlink_to("unknown-format")
    (root / "secret.pt").write_text("weights")
    (root / ".env").write_text("TOKEN=local-only")
    sync(root)
    tracked = git(root, "ls-files", "-z").split(b"\0")
    assert b"code.py" in tracked and b"below-limit" in tracked and b"link-to-large" in tracked
    assert all(name.encode() not in tracked for name in names)
    assert b"secret.pt" not in tracked and b".env" not in tracked
    assert b"data/DATASETS.md" in tracked and b"data/reports/small.json" in tracked
    assert b"data/raw-unknown" not in tracked
    assert all((root / name).exists() for name in names), "Local data was removed"
    assert git(root, "rev-parse", "HEAD") == subprocess.check_output(["git", "--git-dir", str(remote), "rev-parse", "main"])

    with (root / ".git/autopush.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "pending.py").write_text("# lock check\n")
        sync(root)
        assert b"pending.py" not in git(root, "ls-files", "-z").split(b"\0")
    (root / ".git/MERGE_HEAD").write_text("in progress")
    try:
        sync(root)
        raise AssertionError("Sync continued during a merge")
    except RuntimeError as error:
        assert "in-progress" in str(error)
    print("PASS: size limit, escaping, ignored files, local preservation, push, no-op, lock and merge guard")
