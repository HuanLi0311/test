# RNA workspace

RNA data preparation, RibonanzaNet inference, evaluation and fine-tuning scripts.
Dataset sources and caveats are in [data/DATASETS.md](data/DATASETS.md).
The vendored [RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet) source is
from commit `ace422b`; its original license is in `RibonanzaNet/LICENSE`.

## Automatic GitHub sync

This machine runs the user systemd timer `rna-autopush.timer` every 30 seconds.
It stages additions, edits and deletions, commits when files change, and pushes
to `HuanLi0311/test` on branch `main`. It continues after the terminal closes
and starts again at boot through the existing user linger configuration.
Long-running attempts finish before another starts; failures retry on later ticks.
Remote history is preserved; a divergent remote requires manual resolution.

`.gitignore` excludes environments, credentials, downloads, datasets, weights,
caches and logs. It keeps data documentation and small JSON reports.
Before every automatic commit, `scripts/autopush.py` also adds exact ignore paths
for files **at least 50 MiB**, including unknown extensions, and removes ignored
files from the Git index while leaving local files intact. A second check blocks
oversized staged objects if a file grows during staging. `.gitignore` itself
cannot match file sizes; use this script for manual synchronization too.

```bash
# Run once / inspect recent logs / stop or restart automatic sync
/usr/bin/python3 scripts/autopush.py
journalctl --user -u rna-autopush.service -n 30 --no-pager
systemctl --user disable --now rna-autopush.timer
systemctl --user enable --now rna-autopush.timer

# Offline self-check (uses a temporary local repository)
/usr/bin/python3 scripts/test_autopush.py
```

The original nested `RibonanzaNet/.git` metadata is preserved locally under
`.git/local-backups/RibonanzaNet.git` so that its source files can be included
directly in this repository. Local Git configuration holds the SSH key path;
private keys and Git metadata are never included in commits.
