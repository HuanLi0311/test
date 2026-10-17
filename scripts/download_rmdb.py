#!/usr/bin/env python3
"""Download all 1024 RMDB RDAT files without touching the GitHub API.

The API is unauthenticated and rate-limited to 60 req/h; the earlier version
burned through that budget. Release asset URLs follow a fixed pattern, so we
derive them from the manifest (1024 rmdb_ids) and try each tag in turn.

Parallel with per-transfer stall detection. Re-runnable: files already
present at a non-zero size that parses as a complete RDAT are skipped.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "DasLab/rmdb.github.io"
TAGS = ["data-general", "data-rna-structures", "data-riboswitches",
        "data-puzzle", "data-eterna"]
TAG_SUBDIR = {
    "data-general": "general",
    "data-rna-structures": "rna-structures",
    "data-riboswitches": "riboswitches",
    "data-puzzle": "puzzle",
    "data-eterna": "eterna",
}
MANIFEST = "https://rmdb.stanford.edu/manifest.json"
OUT_ROOT = os.path.expanduser("~/test/rna/data/rmdb/rdat")
UA = {"User-Agent": "rmdb-downloader/3.0"}
STALL_SECONDS = 45
WORKERS = 8


def fetch_manifest():
    req = urllib.request.Request(MANIFEST, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def asset_url(tag, vid):
    return f"https://github.com/{REPO}/releases/download/{tag}/{vid}.rdat"


def looks_complete(path):
    """An RDAT starts with a text header containing RDAT_VERSION."""
    if not os.path.exists(path) or os.path.getsize(path) < 64:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(512)
        return b"RDAT_VERSION" in head
    except OSError:
        return False


def stream(url, tmp):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=STALL_SECONDS) as r, open(tmp, "wb") as f:
        last = time.time()
        got = 0
        while True:
            try:
                chunk = r.read(1 << 20)
            except Exception:
                if time.time() - last > STALL_SECONDS:
                    raise IOError("stalled")
                continue
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            last = time.time()
    return got


def download(vid, retries=4):
    for tag in TAGS:
        dest = os.path.join(OUT_ROOT, TAG_SUBDIR[tag], f"{vid}.rdat")
        if looks_complete(dest):
            return "skip"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        for attempt in range(retries):
            try:
                got = stream(asset_url(tag, vid), tmp)
                if got == 0:
                    raise IOError("empty body (wrong tag)")
                os.replace(tmp, dest)
                return "ok"
            except urllib.error.HTTPError as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if e.code == 404:
                    break  # wrong tag, try the next one
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"    {vid}[{tag}] retry {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
    return "fail"


def main():
    manifest = fetch_manifest()
    ids = [e["rmdb_id"] for e in manifest]
    print(f"manifest: {len(ids)} entries", flush=True)

    ok = skip = fail = 0
    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download, vid): vid for vid in ids}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res == "ok":
                ok += 1
            elif res == "skip":
                skip += 1
            else:
                fail += 1
                failed.append(futures[fut])
            if i % 25 == 0:
                print(f"  [{i}/{len(ids)}] ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"ALL DONE ok={ok} skip={skip} fail={fail}", flush=True)
    if failed:
        with open(os.path.expanduser("~/test/rna/logs/rmdb_failed.txt"), "w") as f:
            f.write("\n".join(failed) + "\n")
        print(f"{len(failed)} failures listed in logs/rmdb_failed.txt", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
