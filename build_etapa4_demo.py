"""Ruta rápida: regenera output/etapa4_forecast.json con las 5 estaciones SIN construir
ni ejecutar el notebook completo.

No duplica lógica: toma `etapa4_src.py` (única fuente de verdad) y le aplica un
conjunto acotado de reducciones para que corra en ~10-15 min en vez de ~40:

  - LSTM: 4 épocas en vez de 18 (predicción tosca, pero mantiene la fila en las
    tablas de Diebold-Mariano y heterocedasticidad).
  - rolling-origin: 3 folds en vez de 5.
  - Random Forest / XGBoost: menos árboles.
  - CQR: menos árboles en los GradientBoosting cuantílicos.
  - bootstrap por bloques: 500 réplicas en vez de 2000.

Para el informe formal y los números definitivos:  python build_etapa4.py

    python build_etapa4_demo.py
"""
import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

SRC = Path("etapa4_src.py")

REEMPLAZOS = [
    ("def entrenar_lstm(estacion, epocas=18", "def entrenar_lstm(estacion, epocas=4"),
    ("N_FOLDS = 5", "N_FOLDS = 3"),
    ("reps=2000", "reps=500"),
    ("n_estimators=300, max_depth=16", "n_estimators=160, max_depth=14"),
    ("n_estimators=500, max_depth=6", "n_estimators=250, max_depth=5"),
    ("n_estimators=150, max_depth=3, learning_rate=0.05",
     "n_estimators=90, max_depth=3, learning_rate=0.05"),
    # sin ventanas gráficas ni bloqueo por backend
    ("import matplotlib.pyplot as plt",
     "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt"),
    ("plt.show()", "plt.close('all')"),
]


def main():
    codigo = SRC.read_text(encoding="utf-8")
    for viejo, nuevo in REEMPLAZOS:
        if viejo not in codigo:
            print(f"AVISO: no se encontró para reemplazar: {viejo!r}")
        codigo = codigo.replace(viejo, nuevo)

    tmp = Path("_etapa4_demo_tmp.py")
    tmp.write_text(codigo, encoding="utf-8")
    print("Ejecutando etapa4_src.py en modo rápido (5 estaciones, LSTM 4 épocas)...")
    try:
        runpy.run_path(str(tmp), run_name="__main__")
    finally:
        tmp.unlink(missing_ok=True)
    print("\nListo. output/etapa4_forecast.json regenerado (5 estaciones, ruta rápida).")
    print("Para los números definitivos con LSTM completa: python build_etapa4.py")


if __name__ == "__main__":
    sys.exit(main())
