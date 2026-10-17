#!/usr/bin/env python3
"""Download Kaggle datasets using credentials from ~/.kaggle/kaggle.json.

Uses the v1 REST API directly (no kaggle CLI needed). Handles the
redirect-to-blob that the datasets/download endpoint performs.

Usage:
    download_kaggle.py <owner/dataset> [<owner/dataset> ...]
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

CRED = os.path.expanduser("~/.kaggle/kaggle.json")
OUT_ROOT = os.path.expanduser("~/test/rna/data/kaggle")


def credentials():
    with open(CRED) as f:
        c = json.load(f)
    token = base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}", "User-Agent": "kaggle-fetch/1.0"}


def download(ref, headers):
    owner, name = ref.split("/")
    dest_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(OUT_ROOT, f"{name}.zip")

    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"  zip already present ({os.path.getsize(zip_path)/1e6:.1f} MB), skipping download")
    else:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}"
        print(f"  GET {url}")
        req = urllib.request.Request(url, headers=headers)
        tmp = zip_path + ".part"
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done % (50 << 20) < (1 << 20):
                    print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
        os.replace(tmp, zip_path)
        print(f"  downloaded {os.path.getsize(zip_path)/1e6:.1f} MB")

    print(f"  extracting -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"  files: {len(os.listdir(dest_dir))}")


def main(refs):
    headers = credentials()
    for ref in refs:
        print(f"== {ref}")
        try:
            download(ref, headers)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            print(f"  HTTP {e.code}: {body}")
            if e.code in (401, 403):
                print("  -> check credentials, or accept the competition rules on the web page")
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
