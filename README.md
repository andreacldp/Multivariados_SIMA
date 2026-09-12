# Pronóstico de PM2.5 · Zona Metropolitana de Monterrey

Proyecto de la materia MA2003B: limpieza, exploración, series de tiempo y
modelos de Machine Learning para pronosticar `PM2.5` a una hora, con datos
horarios del SIMA/SIMAJ (2020-2025) para 5 estaciones del Área Metropolitana
de Monterrey.

Usa **dos lenguajes**: R (Etapa 1, carga y primer vistazo de datos, vía
Quarto) y Python (limpieza y Etapas 2-4, el grueso del análisis y los
modelos). El dashboard público del pronóstico vive en [`web/`](web/README.md).

## Orden de los archivos (como se explican en el reporte)

| # | Archivo | Lenguaje | Contenido |
|---|---|---|---|
| 1 | [`quarto/reporte.qmd`](quarto/reporte.qmd) | R (Quarto) | Etapa 1 — conociendo el negocio: carga inicial de los `.xlsx` del Socio Formador |
| 2 | [`limpieza_datos.ipynb`](limpieza_datos.ipynb) | Python | Comprensión y limpieza de datos: valores faltantes, atípicos, centinela, reestructuración; guarda la base limpia en `output/` |
| 3 | [`etapa2.ipynb`](etapa2.ipynb) | Python | Etapa 2 — estadística descriptiva, patrones temporales/espaciales, definición del objetivo |
| 4 | [`etapa3.ipynb`](etapa3.ipynb) (fuente: [`etapa3_src.py`](etapa3_src.py)) | Python | Etapa 3 — SARIMAX, Random Forest, XGBoost, LSTM para NE/SO; primer intervalo de predicción |
| 5 | [`etapa4.ipynb`](etapa4.ipynb) (fuente: [`etapa4_src.py`](etapa4_src.py)) | Python | Etapa 4 — 5 estaciones; error bajo heterocedasticidad/autocorrelación (HAC, bootstrap), rolling-origin, Diebold–Mariano, intervalos condicionales (GARCH, CQR) |

`etapa3.ipynb` y `etapa4.ipynb` **no se editan a mano**: se generan y ejecutan
a partir de `etapa3_src.py` / `etapa4_src.py` (formato `# %%` de celdas) con
`build_etapa3.py` / `build_etapa4.py`. Para reproducir un notebook desde su
fuente:

```bash
python build_etapa4.py            # reconstruye y ejecuta etapa4.ipynb (~25-40 min)
python build_etapa4.py --no-exec  # solo reconstruye la estructura de celdas, sin correrlas
```

> `--no-exec` sirve para revisar que las celdas quedaron bien formadas, pero
> **borra los resultados ya guardados** del notebook (gráficas, tablas
> impresas). No lo corras si solo quieres consultar outputs existentes.

`build_forecast_demo.py` y `build_etapa4_demo.py` son variantes más rápidas
usadas para regenerar `output/etapaN_forecast.json` (los datos que consume el
dashboard en `web/`) sin volver a correr el notebook académico completo.

## Cómo correrlo desde cero

### 1. Dataset (importante — es lo primero)

Los datos crudos (`data/BD 2020.xlsx` … `data/BD 2025.xlsx`) **sí están en el
repo**. La base ya limpia (`output/sima_2020_2025_limpio.parquet`) **no**
—`output/` está en `.gitignore` porque es un artefacto derivado, no una
fuente— así que hay que regenerarla una vez, corriendo `limpieza_datos.ipynb`
de principio a fin. Todo lo demás (`etapa2/3/4`) lee ese parquet y falla con
un `FileNotFoundError` explícito si no existe.

### 2. Entorno Python (limpieza + Etapas 2-4)

```bash
python -m pip install -r requirements.txt
jupyter notebook limpieza_datos.ipynb   # correr primero, de arriba a abajo
jupyter notebook etapa2.ipynb
jupyter notebook etapa3.ipynb
jupyter notebook etapa4.ipynb
```

### 3. Entorno R (Etapa 1 / Quarto)

El proyecto usa `renv` para fijar versiones de paquetes de R:

```r
renv::restore()        # instala las versiones exactas de renv.lock
```

```bash
quarto render quarto/reporte.qmd --to docx   # genera docs/quarto/reporte.docx
```

### 4. Dashboard web (opcional)

Ver [`web/README.md`](web/README.md) — sitio estático en Node puro que
consume el export de `etapa4.ipynb` (`output/etapa4_forecast.json`).

## Estructura

```
data/                 6 archivos .xlsx crudos del SIMA (2020-2025), en el repo
output/               base limpia (.parquet) y exports (.json) — generado localmente, no versionado
quarto/reporte.qmd    Etapa 1 en R
limpieza_datos.ipynb  limpieza de datos (Python)
etapa2.ipynb          Etapa 2 (Python)
etapa3.ipynb / .py    Etapa 3 (Python)
etapa4.ipynb / .py    Etapa 4 (Python)
web/                  dashboard público del pronóstico (Node.js)
renv.lock, .Rprofile  entorno R reproducible (renv)
requirements.txt      entorno Python reproducible
```
