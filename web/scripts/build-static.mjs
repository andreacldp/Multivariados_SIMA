// Genera un HTML autocontenido (datos incrustados, sin servidor) en dist/index.html.
// Útil para previsualizar o compartir el dashboard sin desplegarlo.
//
//   node scripts/build-static.mjs

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const aqui = dirname(fileURLToPath(import.meta.url));
const html = await readFile(resolve(aqui, "../public/index.html"), "utf8");
const datos = JSON.parse(await readFile(resolve(aqui, "../data/forecast.json"), "utf8"));

const inyeccion =
  `<script>window.__FORECAST__=${JSON.stringify(datos).replace(/<\//g, "<\\/")};</script>\n`;

const salida = html.replace('<script>\n"use strict";', inyeccion + '<script>\n"use strict";');
if (salida === html) throw new Error("no se encontró el punto de inyección en index.html");

await mkdir(resolve(aqui, "../dist"), { recursive: true });
await writeFile(resolve(aqui, "../dist/index.html"), salida, "utf8");
console.log("dist/index.html generado (autocontenido,", salida.length, "bytes)");
