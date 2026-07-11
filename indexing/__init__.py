# -*- coding: utf-8 -*-
"""
Indexing package — MS MARCO passage indexing via PyTerrier or PISA.

Use ``get_indexer()`` to obtain the correct indexer for the configured method.
"""

from __future__ import annotations

import config
from indexing.base import BaseIndexer


def get_indexer(
    method: str | None = None, index_dir: str | None = None
) -> BaseIndexer:
    """
    Factory: return the appropriate indexer for the given method.

    Parameters
    ----------
    method : str | None
        ``"pyterrier"`` or ``"pisa"``.  Defaults to ``config.INDEXING_METHOD``.
    index_dir : str | None
        Index directory.  Defaults to ``config.INDEX_DIR``.
    """
    method = method or config.INDEXING_METHOD
    index_dir = index_dir or config.INDEX_DIR

    if method == "pyterrier":
        from indexing.pyterrier_indexer import TerrierIndexer

        return TerrierIndexer(index_dir)
    elif method == "pisa":
        from indexing.pisa_indexer import PisaIndexer

        return PisaIndexer(index_dir)
    else:
        raise ValueError(
            f"INDEXING_METHOD no soportado: '{method}'. "
            "Debe ser 'pyterrier' o 'pisa'."
        )
