"""Gráfica limpia de la serie temporal de PM2.5 para las estaciones NE y SO."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

DATA_DIR = Path("data")
YEARS = range(2020, 2026)
ESTACIONES = ["NE", "SO"]
NOMBRE_LARGO = {"NE": "NE · Noreste (San Nicolás)", "SO": "SO · Suroeste (Santa Catarina)"}
COLOR = {"NE": "#2a78d6", "SO": "#eb6834"}  # slots categóricos 1 y 2 (validados CVD)

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
for est in ESTACIONES:
    d = pd.concat(frames[est], ignore_index=True).dropna(subset=["fecha_hora"])
    d = d.drop_duplicates("fecha_hora").set_index("fecha_hora").sort_index()
    horaria = d["PM2.5"].asfreq("h")
    series_diaria[est] = horaria.resample("D").mean()

# --- gráfica ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
})

fig, ax = plt.subplots(figsize=(12, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")

for est in ESTACIONES:
    s = series_diaria[est]
    roll = s.rolling(30, min_periods=15, center=True).mean()
    ax.plot(s.index, s.values, color=COLOR[est], linewidth=0.6, alpha=0.28, zorder=2)
    ax.plot(roll.index, roll.values, color=COLOR[est], linewidth=2.2,
             label=NOMBRE_LARGO[est], zorder=3, solid_capstyle="round")

fig.suptitle("PM2.5 — promedio diario y tendencia (media móvil 30 días)", fontsize=13,
             color="#0b0b0b", x=0.01, y=0.98, ha="left", fontweight="bold")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.grid(False, axis="x")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="both", length=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylim(bottom=0)

legend = ax.legend(loc="upper left", frameon=False, fontsize=10, ncols=2)

fig.text(0.01, -0.02, "Fuente: SIMA (2020–2025) · línea fina = promedio diario, línea gruesa = media móvil 30 días",
          fontsize=8.5, color="#898781")

fig.tight_layout(rect=(0, 0, 1, 0.94))
out_path = Path("docs/pm25_serie_NE_SO.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Guardado: {out_path.resolve()}")
