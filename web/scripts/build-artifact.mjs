// Genera dist/artifact.html: el mismo dashboard pero sin las etiquetas de documento
// (<!doctype>, <html>, <head>, <body>), con los datos incrustados. Formato apto para
// publicarse como Artifact de Claude (que envuelve el archivo en su propio esqueleto).
//
//   node scripts/build-artifact.mjs

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const aqui = dirname(fileURLToPath(import.meta.url));
let html = await readFile(resolve(aqui, "../public/index.html"), "utf8");
const datos = JSON.parse(await readFile(resolve(aqui, "../data/forecast.json"), "utf8"));

// quedarnos solo con el contenido entre <head> y </body>, sin esas etiquetas
html = html
  .replace(/^[\s\S]*?<head>\s*/i, "")
  .replace(/<\/head>\s*<body>\s*/i, "")
  .replace(/\s*<\/body>\s*<\/html>\s*$/i, "\n");

// el esqueleto del Artifact ya trae charset/viewport y pone su propio favicon
html = html
  .replace(/<meta charset[^>]*>\s*/i, "")
  .replace(/<meta name="viewport"[^>]*>\s*/i, "")
  .replace(/<meta name="description"[^>]*>\s*/i, "")
  .replace(/<link rel="icon"[^>]*>\s*/i, "");

const inyeccion =
  `<script>window.__FORECAST__=${JSON.stringify(datos).replace(/<\//g, "<\\/")};</script>\n`;
html = html.replace('<script>\n"use strict";', inyeccion + '<script>\n"use strict";');

await mkdir(resolve(aqui, "../dist"), { recursive: true });
await writeFile(resolve(aqui, "../dist/artifact.html"), html.trimStart(), "utf8");
console.log("dist/artifact.html generado (", html.length, "bytes )");
