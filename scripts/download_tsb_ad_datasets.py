#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download and extract the TSB-AD datasets into `Datasets/TSB-AD-Datasets`.

Datasets:
  - TSB-AD-U: https://www.thedatum.org/datasets/TSB-AD-U.zip
  - TSB-AD-M: https://www.thedatum.org/datasets/TSB-AD-M.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DATASETS: dict[str, dict[str, str]] = {
    "U": {"name": "TSB-AD-U", "url": "https://www.thedatum.org/datasets/TSB-AD-U.zip"},
    "M": {"name": "TSB-AD-M", "url": "https://www.thedatum.org/datasets/TSB-AD-M.zip"},
}


def _default_target_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Datasets" / "TSB-AD-Datasets"


def _has_any_csv_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(path.suffix.lower() == ".csv" for path in directory.iterdir() if path.is_file())


def _download(url: str, destination: Path, *, timeout: int, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"[skip] Already downloaded: {destination}")
        return

    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    request = Request(url, headers={"User-Agent": "TSB-AD dataset downloader"})
    try:
        with urlopen(request, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            total_size_int = int(total_size) if total_size and total_size.isdigit() else None

            progress = None
            if tqdm is not None and total_size_int is not None:
                progress = tqdm(
                    total=total_size_int,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                )

            with open(temp_path, "wb") as file_handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file_handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))

            if progress is not None:
                progress.close()
    except (HTTPError, URLError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    temp_path.replace(destination)
    print(f"[ok] Downloaded: {destination}")


def _validate_zip_members(zip_file: zipfile.ZipFile) -> None:
    for member in zip_file.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zip_file:
        _validate_zip_members(zip_file)
        zip_file.extractall(extract_dir)


def _move_extracted_dataset(extract_root: Path, dataset_name: str, target_dir: Path, *, force: bool) -> Path:
    dataset_dir = target_dir / dataset_name

    if dataset_dir.exists():
        if not force and _has_any_csv_files(dataset_dir):
            print(f"[skip] Already extracted: {dataset_dir}")
            return dataset_dir
        shutil.rmtree(dataset_dir)

    expected_dir = extract_root / dataset_name
    if expected_dir.is_dir():
        shutil.move(str(expected_dir), str(dataset_dir))
        return dataset_dir

    top_level = list(extract_root.iterdir())
    if len(top_level) == 1 and top_level[0].is_dir():
        shutil.move(str(top_level[0]), str(dataset_dir))
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for item in top_level:
        shutil.move(str(item), str(dataset_dir / item.name))
    return dataset_dir


def _download_and_extract_dataset(
    dataset_key: str,
    target_dir: Path,
    *,
    timeout: int,
    force: bool,
    keep_zips: bool,
) -> None:
    dataset_name = DATASETS[dataset_key]["name"]
    url = DATASETS[dataset_key]["url"]

    downloads_dir = target_dir / "_downloads"
    zip_path = downloads_dir / f"{dataset_name}.zip"

    dataset_dir = target_dir / dataset_name
    if dataset_dir.exists() and not force and _has_any_csv_files(dataset_dir):
        print(f"[skip] {dataset_name} already present at {dataset_dir}")
        return

    _download(url, zip_path, timeout=timeout, force=force)

    with tempfile.TemporaryDirectory(dir=target_dir, prefix=f".tmp_extract_{dataset_name}_") as temp_dir:
        temp_path = Path(temp_dir)
        _extract_zip(zip_path, temp_path)
        final_dir = _move_extracted_dataset(temp_path, dataset_name, target_dir, force=force)
        print(f"[ok] Extracted: {final_dir}")

    if not keep_zips and zip_path.exists():
        zip_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract TSB-AD datasets.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=_default_target_dir(),
        help="Where to store the extracted datasets (default: Datasets/TSB-AD-Datasets).",
    )
    parser.add_argument(
        "--datasets",
        choices=["all", "U", "M"],
        default="all",
        help="Which dataset(s) to download: all, U, or M (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract, overwriting existing data.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep the downloaded zip files after extraction.")
    parser.add_argument("--timeout", type=int, default=60, help="Network timeout in seconds (default: 60).")
    args = parser.parse_args()

    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    selected = ["U", "M"] if args.datasets == "all" else [args.datasets]
    for dataset_key in selected:
        _download_and_extract_dataset(
            dataset_key,
            target_dir,
            timeout=args.timeout,
            force=args.force,
            keep_zips=args.keep_zips,
        )

    print(f"[done] Datasets available under: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

