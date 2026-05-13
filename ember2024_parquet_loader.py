#!/usr/bin/env python3
"""
EMBER2024 Parquet Data Loader Utilities.

Vendored copy of project-wide loader. Handles the 7 canonical tasks:
    load_detection, load_family, load_multilabel, load_challenge_mixed.

Expected parquet filenames in parquet_dir:
    ember2024_train.parquet
    ember2024_test.parquet
    ember2024_challenge.parquet

Each parquet has 2568 feature columns feature_0000..feature_2567 plus metadata.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


FEATURE_COLS = [f"feature_{i:04d}" for i in range(2568)]
TAG_COLUMNS = ["behavior", "file_property", "packer", "exploit", "group"]
IGNORE_TAGS = {"", "win32", "win64", "elf", "linux", "pdf", "apk", "android"}


def load_parquet(
    parquet_dir: str,
    subset: str = "train",
    file_types: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
) -> pl.DataFrame:
    parquet_path = Path(parquet_dir) / f"ember2024_{subset}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pl.read_parquet(parquet_path, columns=columns)

    if file_types is not None:
        file_types_lower = [ft.lower() for ft in file_types]
        df = df.filter(pl.col("file_type").str.to_lowercase().is_in(file_types_lower))
    return df


def extract_features(df: pl.DataFrame) -> np.ndarray:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df.select(available).to_numpy().astype(np.float32)


def load_detection(
    parquet_dir: str,
    subset: str = "train",
    file_types: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    df = load_parquet(parquet_dir, subset, file_types)
    X = extract_features(df)
    y = df["label"].to_numpy().astype(np.int32)
    meta = df.select([c for c in df.columns if not c.startswith("feature_")])
    return X, y, meta


def load_challenge_mixed(
    parquet_dir: str,
    file_types: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame]:
    """Load challenge malware + test benign for mixed evaluation."""
    df_test = load_parquet(parquet_dir, "test", file_types)
    df_test_benign = df_test.filter(pl.col("label") == 0)
    X_test_benign = extract_features(df_test_benign)
    y_test_benign = df_test_benign["label"].to_numpy().astype(np.int32)

    df_challenge = load_parquet(parquet_dir, "challenge", file_types)
    X_challenge = extract_features(df_challenge)
    y_challenge = df_challenge["label"].to_numpy().astype(np.int32)

    meta_tb = df_test_benign.select([c for c in df_test_benign.columns
                                     if not c.startswith("feature_")])
    meta_ch = df_challenge.select([c for c in df_challenge.columns
                                   if not c.startswith("feature_")])
    return X_test_benign, y_test_benign, X_challenge, y_challenge, meta_tb, meta_ch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for subset in ["train", "test", "challenge"]:
        X, y, meta = load_detection(args.parquet_dir, subset,
                                    file_types=["win32", "win64", "dot_net"])
        print(f"{subset}: X={X.shape}, mal={y.sum():,}, ben={(y==0).sum():,}")


if __name__ == "__main__":
    main()
