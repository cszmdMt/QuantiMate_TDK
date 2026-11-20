# -*- coding: utf-8 -*-
"""mainV8.ipynb generálása nbformat-tal (a V7-generátor mintájára).

Futtatás:  python scripts/build_notebook_v8.py
A mainV5/V6/V7.ipynb notebookokat NEM érinti. A generált notebook
output-mentes; kimenetekkel a futtatás (nbconvert --execute) tölti fel.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "generators"))


from nb_walkforward import cells as experiment_cells  # noqa: E402

# Original research cell sources remain in nb_v8*_part*.py.
ALL_CELLS = experiment_cells("V8")


def build() -> nbformat.NotebookNode:
    cells = []
    for i, cell in enumerate(ALL_CELLS):
        kind, src = cell[0], cell[1]
        cid = f"v8-{i:03d}"
        if kind == "md":
            c = new_markdown_cell(src)
        else:
            c = new_code_cell(src)
            tags = cell[2] if len(cell) > 2 else []
            if tags:
                c.metadata["tags"] = tags
        c.id = cid
        cells.append(c)

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "QuantiMate V6 (.venv)",
        "language": "python",
        "name": "quantimate-v6",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12.0"}
    nb.nbformat = 4
    nb.nbformat_minor = 5
    return nb


if __name__ == "__main__":
    nb = build()
    nbformat.validate(nb)
    out = REPO / "experiments" / "notebooks" / "active" / "mainV8.ipynb"
    nbformat.write(nb, out)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"OK: {out.name} kiirva — {len(nb.cells)} cella "
          f"({n_code} kod + {len(nb.cells) - n_code} markdown)")
