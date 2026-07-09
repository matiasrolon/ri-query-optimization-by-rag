# -*- coding: utf-8 -*-
"""
Dataset loading module.
Reads the MS MARCO passage collection from a local TSV file (pid \\t passage).
"""

import csv
import itertools
import os
import time

import config


def load_collection() -> list[dict]:
    """
    Load passages from the local ``collection.tsv`` file.

    The file is expected to live at the path defined by
    ``config.COLLECTION_FILE`` and have two tab-separated columns
    (no header): **pid** and **passage**.

    If ``config.SAMPLE_FRACTION < 1.0``, only the first
    ``config.SAMPLE_SIZE`` passages are loaded.

    Returns
    -------
    list[dict]
        List of document dictionaries with keys ``'docno'`` and ``'text'``.
    """
    collection_path = config.COLLECTION_FILE

    if not os.path.isfile(collection_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de colección: {collection_path}\n"
            "Asegurate de colocar 'collection.tsv' en la carpeta ./data/"
        )

    total = config.SAMPLE_SIZE
    print(f"Muestra objetivo: {total:,} passages ({config.SAMPLE_FRACTION:.0%})")
    print(f"Leyendo colección desde: {collection_path}")

    t0 = time.time()

    docs: list[dict] = []
    with open(collection_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in itertools.islice(reader, total):
            pid, passage = row[0], row[1]
            docs.append({"docno": pid, "text": passage})

    elapsed = time.time() - t0

    print(f"Passages en memoria: {len(docs):,}  (carga: {elapsed:.1f}s)")
    print(f"Ejemplo: {docs[0]}")

    return docs
