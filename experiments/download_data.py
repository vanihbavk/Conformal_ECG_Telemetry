#!/usr/bin/env python3
"""Download and verify the PhysioNet datasets used by ecgrisk into data/."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgrisk.datasets import MITDB_RECORDS_48  # noqa: E402

NSTDB_RECORDS = ("bw", "em", "ma")
CINC11_URL = "https://physionet-open.s3.amazonaws.com/challenge-2011/1.0.0/set-a/"
CINC17_REF_URLS = (
    "https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/training/REFERENCE.csv",
    "https://physionet.org/files/challenge-2017/1.0.0/training/REFERENCE.csv",
)
CINC17_ZIP_URLS = (
    "https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/training2017.zip",
    "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip",
)
CINC17_ZIP_MIN_BYTES = 90_000_000
CINC17_N_RECORDS = 8528

# Peak disk use in MB, used for the free-space check.
PEAK_MB = {"mitdb": 90, "nstdb": 6, "cinc2011": 115, "cinc2017": 255}


def session(workers: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=workers + 4))
    return s


def download(s: requests.Session, urls, dest: Path, min_bytes: int = 1, timeout: int = 60) -> None:
    """Fetch from the first working mirror via a .part file so a failed run leaves no partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    err = None
    for url in urls:
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if part.stat().st_size < min_bytes:
                raise IOError(f"{dest.name}: file too small")
            os.replace(part, dest)
            return
        except Exception as e:
            err = e
            part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {dest.name}: {err}")


def extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError(f"unsafe path in archive: {name}")
        z.extractall(dest)


def present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20


def verify_wfdb(root: Path, name: str, records, exts) -> tuple[bool, str]:
    missing = [f"{r}.{e}" for r in records for e in exts if not present(root / name / f"{r}.{e}")]
    if missing:
        return False, f"{len(missing)} files missing, e.g. {missing[0]}"
    return True, f"{len(records)} records, {size_mb(root / name):.0f} MB"


def verify_mitdb(root: Path):
    return verify_wfdb(root, "mitdb", MITDB_RECORDS_48, ("hea", "dat", "atr"))


def verify_nstdb(root: Path):
    return verify_wfdb(root, "nstdb", NSTDB_RECORDS, ("hea", "dat"))


def cinc11_ids(d: Path) -> list[str]:
    ids = set()
    for name in ("RECORDS-acceptable", "RECORDS-unacceptable"):
        ids.update(l.strip() for l in (d / name).read_text().splitlines() if l.strip())
    return sorted(ids)


def verify_cinc2011(root: Path):
    d = root / "cinc2011"
    if not (present(d / "RECORDS-acceptable") and present(d / "RECORDS-unacceptable")):
        return False, "record lists missing"
    ids = cinc11_ids(d)
    missing = [f"{i}{e}" for i in ids for e in (".hea", ".dat") if not present(d / "set-a" / f"{i}{e}")]
    if missing:
        return False, f"{len(missing)} files missing, e.g. {missing[0]}"
    return True, f"{len(ids)} records, {size_mb(d):.0f} MB"


def verify_cinc2017(root: Path):
    d = root / "cinc2017"
    if not (present(d / "REFERENCE.csv") or present(d / "training2017" / "REFERENCE.csv")):
        return False, "REFERENCE.csv missing"
    n = sum(1 for p in (d / "training2017").glob("*.mat") if p.stat().st_size > 0)
    if n < CINC17_N_RECORDS:
        return False, f"{n}/{CINC17_N_RECORDS} recordings"
    return True, f"{n} records, {size_mb(d):.0f} MB"


def fetch_wfdb(root: Path, name: str, records) -> None:
    import wfdb

    wfdb.dl_database(name, str(root / name), records=list(records))


def fetch_cinc2011(root: Path, s: requests.Session, workers: int, force: bool) -> None:
    d = root / "cinc2011"
    for name in ("RECORDS-acceptable", "RECORDS-unacceptable"):
        if force or not present(d / name):
            download(s, [CINC11_URL + name], d / name, timeout=30)

    def one(rec: str) -> None:
        for ext in (".hea", ".dat"):
            target = d / "set-a" / (rec + ext)
            if force or not present(target):
                download(s, [CINC11_URL + rec + ext], target, timeout=30)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, cinc11_ids(d)))


def fetch_cinc2017(root: Path, s: requests.Session, keep_zip: bool, force: bool) -> None:
    d = root / "cinc2017"
    zip_path = d / "training2017.zip"
    if force or not present(d / "REFERENCE.csv"):
        download(s, CINC17_REF_URLS, d / "REFERENCE.csv", timeout=30)
    if force or not verify_cinc2017(root)[0]:
        if force or not present(zip_path):
            download(s, CINC17_ZIP_URLS, zip_path, min_bytes=CINC17_ZIP_MIN_BYTES, timeout=120)
        extract(zip_path, d)
        if not keep_zip:
            zip_path.unlink()


VERIFY = {"mitdb": verify_mitdb, "nstdb": verify_nstdb, "cinc2011": verify_cinc2011, "cinc2017": verify_cinc2017}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=["all", *VERIFY], default=["all"])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--verify-only", action="store_true", help="check local files, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--keep-zip", action="store_true", help="keep training2017.zip after extraction")
    ap.add_argument("--workers", type=int, default=12, help="parallel downloads for cinc2011")
    args = ap.parse_args()

    root = args.data_dir.resolve()
    names = list(VERIFY) if "all" in args.datasets else list(dict.fromkeys(args.datasets))

    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
        need = sum(PEAK_MB[n] for n in names)
        free = shutil.disk_usage(root).free / 2**20
        if free < need:
            sys.exit(f"need about {need} MB free in {root}, found {free:.0f} MB")

    s = session(args.workers)
    failed = False
    for n in names:
        ok, msg = VERIFY[n](root)
        if not args.verify_only and (args.force or not ok):
            if n == "mitdb":
                fetch_wfdb(root, n, MITDB_RECORDS_48)
            elif n == "nstdb":
                fetch_wfdb(root, n, NSTDB_RECORDS)
            elif n == "cinc2011":
                fetch_cinc2011(root, s, args.workers, args.force)
            else:
                fetch_cinc2017(root, s, args.keep_zip, args.force)
            ok, msg = VERIFY[n](root)
        print(f"{n}: {'ok' if ok else 'FAILED'} ({msg})")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
