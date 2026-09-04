// Copia el último export del notebook (../output/etapa3_forecast.json)
// a web/data/forecast.json, que es lo que sirve la función /api/forecast.
//
//   node scripts/sync-data.mjs
//   npm run sync

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const aqui = dirname(fileURLToPath(import.meta.url));
const origen = resolve(aqui, "../../output/etapa3_forecast.json");
const destino = resolve(aqui, "../data/forecast.json");

try {
  const contenido = await readFile(origen, "utf8");
  const datos = JSON.parse(contenido); // valida que sea JSON
  await writeFile(destino, JSON.stringify(datos, null, 2) + "\n", "utf8");
  const n = Object.keys(datos.estaciones ?? {}).length;
  console.log(`sync OK: ${n} estaciones, generado ${datos.generado}`);
  console.log(`  ${origen}`);
  console.log(`  -> ${destino}`);
} catch (err) {
  console.error("No se pudo sincronizar:", err.message);
  console.error("¿Ya ejecutaste etapa3.ipynb? Debe existir output/etapa3_forecast.json");
  process.exit(1);
}
