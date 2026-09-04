"""Regenera output/etapa3_forecast.json con los campos que consume la página web.

Versión rápida para la demo: reproduce EXACTAMENTE la construcción de features y el
Random Forest de `etapa3_src.py` (secciones 4 y 7), calcula los residuos del
conjunto de prueba y añade el bloque de diagnóstico que pide la presentación
(error por nivel, autocorrelación de residuos, homocedasticidad, importancia de
variables, naive vs. RF en cambios de régimen y el peor episodio del periodo).

No reentrena LSTM/SARIMAX: los números de SARIMAX se copian del JSON previo si
existe. Para el informe formal, `python build_etapa3.py` sigue siendo la fuente.

    python build_forecast_demo.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, het_arch, acorr_ljungbox,
)
from statsmodels.tsa.stattools import acf

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

DATA_PATH = Path("output/sima_2020_2025_limpio.parquet")
OUT_PATH = Path("output/etapa3_forecast.json")

TARGET = "PM2.5"
ESTACIONES = ["NE", "SO"]
METEO = ["TOUT", "RH", "SR", "WSR", "WDR", "RAINF", "PRS"]
CONTEXTO = ["PM10"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
VENTANAS = [24, 168]
DIAS_PRUEBA = 60

NOMBRE_LARGO = {"NE": "Noreste · San Nicolás", "SO": "Suroeste · Santa Catarina"}
NIVELES_ES = ["bajo", "medio", "alto", "muy_alto"]

df = pd.read_parquet(DATA_PATH)
col_est = next(c for c in df.columns if c.startswith("estaci"))
df = df.rename(columns={col_est: "estación"})
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])


def construir_serie(estacion, variables):
    d = (df.loc[df["estación"] == estacion, ["fecha_hora"] + variables]
           .set_index("fecha_hora"))
    d = d[~d.index.duplicated(keep="first")].sort_index()
    idx = pd.date_range(d.index.min(), d.index.max(), freq="h")
    return d.reindex(idx)


series_horarias = {est: construir_serie(est, [TARGET] + METEO + CONTEXTO) for est in ESTACIONES}


def construir_features(estacion):
    d = series_horarias[estacion]
    X = pd.DataFrame(index=d.index)
    for L in LAGS:
        X[f"{TARGET}_lag{L}h"] = d[TARGET].shift(L)
    for W in VENTANAS:
        X[f"{TARGET}_media{W}h"] = d[TARGET].shift(1).rolling(W).mean()
        X[f"{TARGET}_std{W}h"] = d[TARGET].shift(1).rolling(W).std()
    h = X.index.hour.to_numpy()
    m = X.index.month.to_numpy()
    X["hora"] = h
    X["dia_semana"] = X.index.dayofweek
    X["mes"] = m
    X["hora_sin"] = np.sin(2 * np.pi * h / 24)
    X["hora_cos"] = np.cos(2 * np.pi * h / 24)
    X["mes_sin"] = np.sin(2 * np.pi * m / 12)
    X["mes_cos"] = np.cos(2 * np.pi * m / 12)
    for v in METEO + CONTEXTO:
        X[f"{v}_lag1h"] = d[v].shift(1)
    y = d[TARGET]
    tabla = pd.concat([X, y.rename("y")], axis=1).dropna()
    return tabla.drop(columns="y"), tabla["y"]


def dividir_temporal(X, y, dias=DIAS_PRUEBA):
    h = dias * 24
    return X.iloc[:-h], X.iloc[-h:], y.iloc[:-h], y.iloc[-h:]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def bloque_bootstrap_mae(y_real, y_pred, indice, reps=1000):
    dfb = pd.DataFrame({"abs": np.abs(np.asarray(y_real, float) - np.asarray(y_pred, float))},
                       index=pd.DatetimeIndex(indice))
    grupos = [g for _, g in dfb.groupby(dfb.index.normalize())]
    n = len(grupos)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = pd.concat([grupos[k] for k in rng.integers(0, n, n)])["abs"].mean()
    return np.percentile(out, [2.5, 97.5])


sarimax_previo = {}
if OUT_PATH.exists():
    try:
        prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        for k, e in prev.get("estaciones", {}).items():
            if "sarimax" in e:
                sarimax_previo[k] = e["sarimax"]
    except Exception:
        pass

export = {
    "generado": pd.Timestamp.now().isoformat(timespec="seconds"),
    "contaminante": TARGET,
    "unidad": "µg/m³",
    "horizonte": "t+1 (1 hora)",
    "fuente": "SIMA / SIMAJ 2020-2025 (base limpia del proyecto)",
    "particion_prueba_dias": DIAS_PRUEBA,
    "estaciones": {},
}

for est in ESTACIONES:
    X, y = construir_features(est)
    X_tr, X_te, y_tr, y_te = dividir_temporal(X, y)
    idx = y_te.index
    real = y_te.to_numpy(float)

    pred_naive = X_te[f"{TARGET}_lag1h"].to_numpy(float)
    rf = RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=3,
                               random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    pred_xgb = xgb.predict(X_te)

    candidatos = {"Naive (t-1)": pred_naive, "Random Forest": pred_rf, "XGBoost": pred_xgb}
    nombre = min(candidatos, key=lambda k: mae(real, candidatos[k]))
    pred = np.asarray(candidatos[nombre], float)
    modelo_arbol = rf if nombre != "XGBoost" else xgb

    e = real - pred                         # residuo = real - pronóstico
    res = pd.DataFrame({"real": real, "pred": pred, "e": e}, index=idx)

    # --- métricas base ---
    mae_v = float(np.mean(np.abs(e)))
    rmse_v = float(np.sqrt(np.mean(e ** 2)))
    sesgo_v = float(np.mean(e))
    mae_naive = mae(real, pred_naive)
    ic_lo, ic_hi = bloque_bootstrap_mae(real, pred, idx)
    q_lo, q_hi = np.percentile(e, [2.5, 97.5])

    # --- error por nivel observado ---
    niv = pd.qcut(res["real"], 4, labels=NIVELES_ES)
    err_nivel = [{"nivel": n,
                  "mae": round(float(res.loc[niv == n, "e"].abs().mean()), 2),
                  "sesgo": round(float(res.loc[niv == n, "e"].mean()), 2)}
                 for n in NIVELES_ES]

    # --- intervalos empíricos: residuos de 30 días de validación antes de la prueba ---
    h_te = DIAS_PRUEBA * 24
    h_val = 30 * 24
    X_val, y_val = X.iloc[-(h_te + h_val):-h_te], y.iloc[-(h_te + h_val):-h_te]
    pred_val = X_val[f"{TARGET}_lag1h"].to_numpy(float) if nombre == "Naive (t-1)" else modelo_arbol.predict(X_val)
    resid_val = y_val.to_numpy(float) - pred_val
    qv = {p: np.percentile(resid_val, p) for p in (2.5, 10, 90, 97.5)}
    cob95 = float(((real >= pred + qv[2.5]) & (real <= pred + qv[97.5])).mean())
    cob80 = float(((real >= pred + qv[10]) & (real <= pred + qv[90])).mean())

    # --- diagnóstico de residuos ---
    dw = float(durbin_watson(e))
    lb = acorr_ljungbox(e, lags=[1, 24, 48, 168], return_df=True)
    ljung = [{"lag": int(l), "p": round(float(r["lb_pvalue"]), 4)} for l, r in lb.iterrows()]
    acf_res = acf(e, nlags=36, fft=False)
    acf_out = [round(float(v), 3) for v in acf_res]           # índice = lag (0..36)
    exog_bp = sm.add_constant(np.column_stack([pred, pred ** 2]))
    bp_p = float(het_breuschpagan(e, exog_bp)[1])
    wh_p = float(het_white(e, exog_bp)[1])
    arch_p = float(het_arch(e, nlags=24)[1])
    homoced = "NO" if min(bp_p, wh_p, arch_p) <= 0.05 else "sí"

    # --- importancia de variables (top 12) ---
    imp = (pd.Series(modelo_arbol.feature_importances_, index=X_tr.columns)
           .sort_values(ascending=False).head(12))
    importancia = [{"var": k, "imp": round(float(v), 4)} for k, v in imp.items()]

    # --- naive vs. modelo en horas de cambio de régimen ---
    delta = np.abs(np.diff(np.concatenate([[real[0]], real])))   # |PM2.5(t) - PM2.5(t-1)|
    umbral = float(np.percentile(delta, 75))
    cambio = delta >= umbral
    nv = {
        "umbral_cambio": round(umbral, 2),
        "n_horas_cambio": int(cambio.sum()),
        "mae_modelo_cambio": round(mae(real[cambio], pred[cambio]), 2),
        "mae_naive_cambio": round(mae(real[cambio], pred_naive[cambio]), 2),
        "mae_modelo_estable": round(mae(real[~cambio], pred[~cambio]), 2),
        "mae_naive_estable": round(mae(real[~cambio], pred_naive[~cambio]), 2),
    }
    nv["mejora_cambio_pct"] = round((nv["mae_naive_cambio"] - nv["mae_modelo_cambio"]) /
                                    nv["mae_naive_cambio"] * 100, 1) if nv["mae_naive_cambio"] else None

    # --- peor episodio: bloque continuo (>= 48 h) con el pico real más alto,
    #     ventana de hasta 72 h alrededor del pico. Se ignoran los picos aislados
    #     de pocas horas (p. ej. pirotecnia de fin de año), que no son episodios
    #     meteorológicos de acumulación.
    serie = res.reset_index(names="fecha_hora")
    serie["consec"] = (serie["fecha_hora"].diff() != pd.Timedelta(hours=1)).cumsum()
    tam = serie.groupby("consec")["real"].transform("size")
    cand = serie[tam >= 48]
    if cand.empty:
        cand = serie
    bloque_id = cand.loc[cand["real"].idxmax(), "consec"]
    bloque = serie[serie["consec"] == bloque_id].reset_index(drop=True)
    pos = int(bloque["real"].idxmax())
    ini = max(0, pos - 48)
    ventana = bloque.iloc[ini:ini + 72].reset_index(drop=True)
    pico_real = float(ventana["real"].max())
    j_pico = int(ventana["real"].idxmax())
    pron_en_pico = float(ventana["pred"].iloc[j_pico])
    # retraso: horas hasta que el pronóstico alcanza el 90 % del pico real
    despues = ventana["pred"].iloc[j_pico:]
    alcanza = np.where(despues.to_numpy() >= 0.9 * pico_real)[0]
    retraso = int(alcanza[0]) if len(alcanza) else None
    episodio = {
        "inicio": ventana["fecha_hora"].iloc[0].isoformat(),
        "fin": ventana["fecha_hora"].iloc[-1].isoformat(),
        "horas": int(len(ventana)),
        "pico_real": round(pico_real, 2),
        "hora_pico": ventana["fecha_hora"].iloc[j_pico].isoformat(),
        "pronostico_en_pico": round(pron_en_pico, 2),
        "subestimacion_pico": round(pico_real - pron_en_pico, 2),
        "retraso_h": retraso,
        "banda95": [round(float(q_lo), 2), round(float(q_hi), 2)],
        "serie": [{"fecha_hora": t.isoformat(), "real": round(float(r), 2),
                   "pronostico": round(float(p), 2)}
                  for t, r, p in zip(ventana["fecha_hora"], ventana["real"], ventana["pred"])],
    }

    # --- serie reciente (últimas 72 observaciones del periodo de prueba) ---
    ultimo = res.iloc[-1]
    serie_reciente = [
        {"fecha_hora": t.isoformat(), "real": round(float(r.real), 2),
         "pronostico": round(float(r.pred), 2)}
        for t, r in res.iloc[-72:].iterrows()
    ]

    export["estaciones"][est] = {
        "clave": est,
        "nombre": NOMBRE_LARGO[est],
        "modelo": nombre,
        "mae": round(mae_v, 2),
        "mae_ic95": f"[{ic_lo:.2f}, {ic_hi:.2f}]",
        "mae_naive": round(mae_naive, 2),
        "rmse": round(rmse_v, 2),
        "sesgo": round(sesgo_v, 2),
        "n_prueba": int(len(res)),
        "periodo_prueba": [res.index.min().isoformat(), res.index.max().isoformat()],
        "banda95": [round(float(q_lo), 2), round(float(q_hi), 2)],
        "cobertura95": round(cob95, 3),
        "cobertura80": round(cob80, 3),
        "homocedastico": homoced,
        "ljung_box_p_lag24": round(float(lb.loc[24, "lb_pvalue"]), 4),
        "sarimax": sarimax_previo.get(est, {"mae_diario": None, "cobertura_ic95": None}),
        "error_por_nivel": err_nivel,
        "importancia_variables": importancia,
        "diagnostico": {
            "durbin_watson": round(dw, 3),
            "ljung_box": ljung,
            "acf_residuos": acf_out,
            "breusch_pagan_p": round(bp_p, 4),
            "white_p": round(wh_p, 4),
            "arch_p": round(arch_p, 4),
            "homocedastico": homoced,
            "n": int(len(e)),
        },
        "naive_vs_modelo": nv,
        "episodio": episodio,
        "ultimo_punto": {
            "fecha_hora": ultimo.name.isoformat(),
            "real": round(float(ultimo["real"]), 2),
            "pronostico": round(float(ultimo["pred"]), 2),
        },
        "serie_reciente": serie_reciente,
    }
    print(f"{est}: campeón={nombre}  MAE={mae_v:.2f} (naive {mae_naive:.2f})  "
          f"DW={dw:.2f}  LB24 p={lb.loc[24, 'lb_pvalue']:.1e}  homoced={homoced}  "
          f"pico={pico_real:.0f} pron={pron_en_pico:.0f} retraso={retraso}")

OUT_PATH.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nExportado: {OUT_PATH}")
