# -*- coding: utf-8 -*-
"""
Shared utilities for indexing modules.
"""

import pathlib


def dir_mb(path: str) -> float:
    """Calculate the total size of a directory in megabytes."""
    return sum(
        f.stat().st_size
        for f in pathlib.Path(path).rglob("*")
        if f.is_file()
    ) / 1e6


def print_stats(
    engine_name: str,
    time_secs: float,
    num_docs: int,
    num_terms: int,
    index_path: str,
    sample_size: int,
    num_tokens: int | None = None,
) -> None:
    """Print standardised indexing statistics."""
    print(f"\n===== {engine_name} =====")
    print(f"Tiempo de indexacion : {time_secs:8.1f} s")
    print(f"Documentos           : {num_docs:,}")
    print(f"Terminos unicos      : {num_terms:,}")
    if num_tokens is not None:
        print(f"Tokens               : {num_tokens:,}")
    print(f"Tamano en disco      : {dir_mb(index_path):.1f} MB")
    print(f"Throughput           : {sample_size / time_secs:,.0f} docs/s")
