#!/usr/bin/env python3
"""Download processed/normalised reactivity data from GEO supplementary files.

Only the small, directly-usable files are fetched here. RAW.tar bundles
(hundreds of MB of fastq) are opt-in via --raw, because they need a
reprocessing pipeline before they yield reactivity.

Re-runnable: existing files with the expected size are skipped.
"""
import argparse
import os
import sys
import time
import urllib.request

OUT_ROOT = os.path.expanduser("~/test/rna/data/geo")
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/{stamp}/{gse}/suppl/{name}"
UA = {"User-Agent": "geo-react-dl/1.0"}

# (accession, [filenames])
PROCESSED = [
    ("GSE153851", [
        "GSE153851_deep234plus_1_29882_popavg_reacts_nobv.txt.gz",
        "GSE153851_sars30kb_deep234_SARS2MN985325WA_plus_combined.ct.gz",
        "GSE153851_sars30kb_deep234_SARS2MN985325WA_plus_combined.dot.gz",
    ]),
    ("GSE132099", [
        "GSE132099_icSHAPE_invivo.out.txt.gz",
        "GSE132099_icSHAPE_invitro.out.txt.gz",
        "GSE132099_icLASER_invivo.out.txt.gz",
        "GSE132099_icLASER_invitro.out.txt.gz",
    ]),
    ("GSE146952", [
        "GSE146952_sRNA_icSHAPE-MaP_invivo_shape.txt.gz",
        "GSE146952_sRNA_icSHAPE-MaP_invitro_shape.txt.gz",
        "GSE146952_RIP_icSHAPE-MaP_shape.txt.gz",
    ]),
    ("GSE236224", [
        "GSE236224_Normalized_reactivity.csv.gz",
    ]),
    ("GSE45803", [
        "GSE45803_Feb13_VivoAllextra_1_15_PLUS.wig.gz",
        "GSE45803_Feb13_VivoAllextra_1_15_Minus.wig.gz",
        "GSE45803_Feb13_VitroAll_1_15_Plus.wig.gz",
        "GSE45803_Feb13_VitroAll_1_15_Minus.wig.gz",
        "GSE45803_Feb13_DenaturedAll_1_15_Plus.wig.gz",
        "GSE45803_Feb13_DenaturedAll_1_15_MINUS.wig.gz",
    ]),
]

RAW = [
    ("GSE145805", "GSE145805_RAW.tar", 445),
    ("GSE117840", "GSE117840_RAW.tar", 108),
    ("GSE45803", "GSE45803_RAW.tar", 384),
]


def stamp(gse):
    return gse[:len(gse) - 3] + "nnn"


def fetch(url, dest, retries=4):
    if os.path.exists(dest):
        return "skip"
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:
            print(f"    retry {attempt + 1}/{retries}: {e}", flush=True)
            time.sleep(4 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="also fetch RAW.tar bundles")
    args = ap.parse_args()

    # ("gse", "filename") for processed files, ("RAW", name, size) for bundles
    jobs = [("gse", gse, name) for gse, names in PROCESSED for name in names]
    if args.raw:
        jobs += [("RAW", name, size) for _, name, size in RAW]

    ok = skip = fail = 0
    for item in jobs:
        if item[0] == "RAW":
            dest = os.path.join(OUT_ROOT, "raw", item[1])
            url = None
            for gse, name, _ in RAW:
                if name == item[1]:
                    url = BASE.format(stamp=stamp(gse), gse=gse, name=name)
                    break
        else:
            _, gse, name = item
            dest = os.path.join(OUT_ROOT, gse, name)
            url = BASE.format(stamp=stamp(gse), gse=gse, name=name)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"== {name}", flush=True)
        res = fetch(url, dest)
        if res == "ok":
            print(f"   {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
            ok += 1
        elif res == "skip":
            print("   already present", flush=True)
            skip += 1
        else:
            print(f"   FAILED {url}", flush=True)
            fail += 1

    print(f"DONE ok={ok} skip={skip} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
