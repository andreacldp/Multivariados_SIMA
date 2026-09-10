// Copia el último export del notebook (../output/etapa4_forecast.json)
// a web/data/forecast.json, que es lo que sirve la función /api/forecast.
// El JSON de Etapa 4 es un superset del de Etapa 3: si aún no existe, cae al de Etapa 3.
//
//   node scripts/sync-data.mjs
//   npm run sync

import { readFile, writeFile, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const aqui = dirname(fileURLToPath(import.meta.url));
const e4 = resolve(aqui, "../../output/etapa4_forecast.json");
const e3 = resolve(aqui, "../../output/etapa3_forecast.json");
const origen = await access(e4).then(() => e4, () => e3);
const destino = resolve(aqui, "../data/forecast.json");

try {
  const contenido = await readFile(origen, "utf8");
  const datos = JSON.parse(contenido); // valida que sea JSON
  await writeFile(destino, JSON.stringify(datos, null, 2) + "\n", "utf8");
  const n = Object.keys(datos.estaciones ?? {}).length;
  console.log(`sync OK: etapa ${datos.etapa ?? 3}, ${n} estaciones, generado ${datos.generado}`);
  console.log(`  ${origen}`);
  console.log(`  -> ${destino}`);
} catch (err) {
  console.error("No se pudo sincronizar:", err.message);
  console.error("¿Ya ejecutaste el notebook? Debe existir output/etapa4_forecast.json (o etapa3_forecast.json).");
  process.exit(1);
}
