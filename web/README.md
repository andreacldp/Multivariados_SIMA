# Pronóstico PM2.5 · ZMM — dashboard ejecutivo

Dashboard del pronóstico de PM2.5 a una hora para las estaciones **NE** (Noreste,
San Nicolás) y **SO** (Suroeste, Santa Catarina) de la Zona Metropolitana de
Monterrey. Los datos y el modelo salen del notebook `../etapa3.ipynb`.

Proyecto Node.js puro (sin dependencias): un sitio estático en `public/` y un
handler en `api/` que sirve el pronóstico. Todo corre con Node local.

## Contenido del dashboard

- **Ahora → en 1 hora**: valor medido, pronóstico, categoría del Índice AIRE y
  SALUD y tendencia (mejora / estable / empeora).
- **Recomendaciones de salud**: mensajes por categoría para población general y
  grupos sensibles, con acciones sugeridas (base: NOM-172).
- **Medición vs. pronóstico** (72 observaciones) + tarjeta de confiabilidad
  (MAE con IC 95%, RMSE, sesgo, mejora vs. ingenuo, cobertura del intervalo).
- **Resumen de 72 observaciones**: promedio, máximo/mínimo, horas en “Buena” y
  en “Mala o peor”.
- **Comparación entre estaciones**.
- **Diagnóstico del modelo · ¿cuándo se equivoca?**: MAE y sesgo por nivel de
  contaminación, factor de degradación del error, y dónde aporta el modelo frente
  a repetir el último valor (horas de cambio de régimen).
- **El peor episodio del periodo de prueba**: serie real vs. pronóstico en la
  ventana continua de 72 h con el pico más alto, con la subestimación y el
  retraso en el pico.
- **Ficha técnica · residuos**: homocedasticidad (Breusch–Pagan, White, ARCH),
  autocorrelación (Durbin–Watson, Ljung–Box) y ACF de los residuos.
- **Cómo modelamos**: importancia de variables del modelo campeón y ficha de
  método (para el equipo técnico de SIMA).

## Estructura

```
web/
├── public/index.html      Dashboard (HTML/CSS/JS sin dependencias; consume /api/forecast)
├── api/forecast.js         Handler Node que devuelve el pronóstico
├── data/forecast.json      Snapshot del export del notebook (se versiona)
├── scripts/dev.mjs         Servidor local con Node puro (sin instalar nada)
├── scripts/sync-data.mjs   Copia ../output/etapa3_forecast.json → data/forecast.json
└── package.json
```

## Uso local

```bash
cd web
npm run dev          # http://localhost:3000  (no requiere npm install)
```

`npm run dev` levanta un servidor con Node puro que sirve `public/` y responde
`/api/forecast` con el handler de `api/forecast.js`.

## Actualizar los datos

Después de volver a ejecutar el notebook (`python ../build_etapa3.py`):

```bash
npm run sync         # refresca data/forecast.json desde ../output/etapa3_forecast.json
```

Atajo para la demo (regenera el export en ~2 min, sin reentrenar LSTM/SARIMAX):

```bash
python ../build_forecast_demo.py   # reescribe ../output/etapa3_forecast.json
npm run sync && npm run static && npm run artifact
```

## Previsualizar sin servidor

```bash
npm run static       # dist/index.html: HTML autocontenido, se abre con doble clic
npm run artifact     # dist/artifact.html: mismo dashboard en formato Claude Artifact
```

## Siguiente iteración

- En `api/forecast.js`, sustituir la lectura del JSON por una consulta a la API en
  vivo de SIMA + inferencia del modelo entrenado (hoy se sirve el último export).
- Intervalos de predicción por nivel de concentración (quantile regression / CQR)
  para corregir la sub-cobertura en episodios altos.
