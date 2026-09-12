"""Exporta CSV resumido de PM2.5 (NE y SO): promedio diario y media móvil 30 días.

Pensado para subir a un chat de Claude sin acceso a disco local (p. ej. para
generar un diseño en Canva a partir de la serie), reduciendo el tamaño de datos
respecto de los Excel originales.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
YEARS = range(2020, 2026)
ESTACIONES = ["NE", "SO"]

frames = {est: [] for est in ESTACIONES}
for year in YEARS:
    path = DATA_DIR / f"BD {year}.xlsx"
    sheets = pd.read_excel(path, sheet_name=ESTACIONES, engine="openpyxl")
    for est in ESTACIONES:
        raw = sheets[est].rename(columns={"Fecha y hora": "fecha_hora", "date": "fecha_hora"})
        d = raw[["fecha_hora", "PM2.5"]].copy()
        d = d[d["fecha_hora"].notna()].copy()
        d["fecha_hora"] = pd.to_datetime(d["fecha_hora"], errors="coerce")
        d["PM2.5"] = pd.to_numeric(d["PM2.5"], errors="coerce")
        frames[est].append(d)

series_diaria = {}
series_roll = {}
for est in ESTACIONES:
    d = pd.concat(frames[est], ignore_index=True).dropna(subset=["fecha_hora"])
    d = d.drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index()
    horaria = d["PM2.5"].asfreq("h")
    diaria = horaria.resample("D").mean()
    series_diaria[est] = diaria
    series_roll[est] = diaria.rolling(30, min_periods=15, center=True).mean()

out = pd.DataFrame({
    "fecha": series_diaria["NE"].index,
    "NE_diario": series_diaria["NE"].values,
    "NE_media_movil_30d": series_roll["NE"].values,
    "SO_diario": series_diaria["SO"].values,
    "SO_media_movil_30d": series_roll["SO"].values,
})
out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
out = out.round(2)

out_path = Path("docs/pm25_serie_NE_SO.csv")
out_path.parent.mkdir(exist_ok=True)
out.to_csv(out_path, index=False)
print(f"Guardado: {out_path.resolve()}")
print(f"Filas: {len(out)}")
