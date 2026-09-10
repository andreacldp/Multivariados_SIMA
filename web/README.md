# Pronóstico PM2.5 · ZMM — dashboard ejecutivo

Dashboard del pronóstico de PM2.5 a una hora para **5 estaciones** del SIMA en la
Zona Metropolitana de Monterrey: **NE** (Noreste, San Nicolás), **SO** (Suroeste,
Santa Catarina), **CE** (Centro), **NTE2** (Norte) y **SE3** (Sureste). Los datos
y el modelo salen del notebook `../etapa4.ipynb` (`../output/etapa4_forecast.json`);
si ese export no existe, `sync-data` cae al de `../etapa3.ipynb`.

Proyecto Node.js puro (sin dependencias): un sitio estático en `public/` y un
handler en `api/` que sirve el pronóstico. Todo corre con Node local.

## Contenido del dashboard

El dashboard es **solo la vista pública** (sin jerga técnica). El análisis
académico completo de la Etapa 4 —cálculo del error (HAC / bootstrap / n
efectivo), heterocedasticidad (BP, White, ARCH, Goldfeld–Quandt), validación
rolling-origin, Diebold–Mariano, intervalos GARCH/CQR— vive en `../etapa4.ipynb`.

- **Titular en una frase**: conclusión ya “masticada” (p. ej. *“Puedes hacer tu
  vida al aire libre con normalidad”*) según la categoría del pronóstico.
- **Ubicación y compartir**: botón para detectar la estación más cercana por
  geolocalización del navegador (con permiso) y botón para compartir el resumen
  (Web Share API, con WhatsApp como respaldo).
- **Ahora → en 1 hora**: valor medido, pronóstico, categoría del Índice AIRE y
  SALUD y tendencia (mejora / estable / empeora).
- **¿Qué significa este número para ti?**: comparación del pronóstico con la guía
  de la OMS (15 µg/m³) y el límite mexicano NOM-025 (41 µg/m³), y si el aire
  viene mejorando o empeorando en la racha de horas seguidas más reciente.
- **¿Es normal?**: posición del pronóstico frente a las 72 observaciones
  recientes (*“peor que N de cada 10”*).
- **Recomendaciones de salud**: mensajes por categoría para población general y
  grupos sensibles, con acciones sugeridas (base: NOM-172), con iconos.
- **Cómo viene el aire · últimas horas**: medición vs. pronóstico (72
  observaciones) con el rango probable, y una tarjeta *“¿se le puede creer al
  pronóstico?”* (semáforo Alta / Media / Baja + cómo usarlo).
- **Resumen de las últimas horas**: promedio, máximo/mínimo, horas en “Buena” y
  en “Mala o peor”.
- **Comparación entre estaciones**: categoría, valor, nivel habitual, rango
  probable y confianza del pronóstico por estación.

## Estructura

```
web/
├── public/index.html      Dashboard (HTML/CSS/JS sin dependencias; consume /api/forecast)
├── api/forecast.js         Handler Node que devuelve el pronóstico
├── data/forecast.json      Snapshot del export del notebook (se versiona)
├── scripts/dev.mjs         Servidor local con Node puro (sin instalar nada)
├── scripts/sync-data.mjs   Copia ../output/etapa4_forecast.json → data/forecast.json
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

Después de volver a ejecutar el notebook (`python ../build_etapa4.py`, ~25-40 min:
LSTM ×5, rolling-origin y CQR):

```bash
npm run sync         # refresca data/forecast.json desde ../output/etapa4_forecast.json
npm run static && npm run artifact
```

`build_etapa4.py` necesita `arch` además de las dependencias de Etapa 3
(`pip install -r ../requirements.txt`).

## Previsualizar sin servidor

```bash
npm run static       # dist/index.html: HTML autocontenido, se abre con doble clic
npm run artifact     # dist/artifact.html: mismo dashboard en formato Claude Artifact
```

## Siguiente iteración

- En `api/forecast.js`, sustituir la lectura del JSON por una consulta a la API en
  vivo de SIMA + inferencia del modelo entrenado (hoy se sirve el último export).
- Los intervalos condicionales (GARCH / CQR) de la Etapa 4 corrigen la
  sub-cobertura en episodios altos a costa de un intervalo más ancho; falta
  llevar el `σ_t` de un solo paso a producción y afinar el ancho con CQR por
  nivel.
- Incorporar la LSTM al rolling-origin (hoy solo entra en el holdout final por
  coste de cómputo).
