"""Convierte etapa4_src.py (formato percent '# %%') en etapa4.ipynb y lo ejecuta.

Uso:
    python build_etapa4.py            # construye y ejecuta el notebook
    python build_etapa4.py --no-exec  # solo construye (sin ejecutar celdas)

Dependencias extra respecto a Etapa 3: `arch` (GARCH). `pip install arch`.
La ejecución completa tarda ~25-40 min (5 estaciones x LSTM + rolling-origin + CQR).
"""
import re
import sys
from pathlib import Path

import nbformat

SRC = Path("etapa4_src.py")
OUT = Path("etapa4.ipynb")


def parse_cells(text):
    partes = re.split(r"(?m)^# %%(.*)$", text)
    celdas = []
    for i in range(1, len(partes), 2):
        marcador = partes[i].strip()
        cuerpo = partes[i + 1].strip("\n")
        if marcador.startswith("[markdown]"):
            m = re.match(r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')\s*$', cuerpo, re.S)
            contenido = (m.group(1) if m else cuerpo).strip("\n")
            celdas.append(nbformat.v4.new_markdown_cell(contenido))
        else:
            celdas.append(nbformat.v4.new_code_cell(cuerpo))
    return celdas


def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = parse_cells(SRC.read_text(encoding="utf-8"))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbformat.write(nb, OUT)
    print(f"Construido {OUT} con {len(nb.cells)} celdas")

    if "--no-exec" not in sys.argv:
        from nbclient import NotebookClient
        print("Ejecutando el notebook (25-40 min: LSTM x5, rolling-origin, CQR)...")
        NotebookClient(nb, timeout=5400, kernel_name="python3").execute()
        nbformat.write(nb, OUT)
        print(f"Ejecutado y guardado {OUT}")


if __name__ == "__main__":
    main()
