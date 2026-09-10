// Servidor local con Node puro (sin dependencias).
// Sirve web/public/ como sitio estático y enruta /api/forecast al handler
// de api/forecast.js.
//
//   node scripts/dev.mjs        (o: npm run dev)
//   http://localhost:3000

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve, extname, normalize } from "node:path";

const aqui = dirname(fileURLToPath(import.meta.url));
const raizPublica = resolve(aqui, "../public");
const PUERTO = process.env.PORT || 3000;

const TIPOS = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const { default: forecastHandler } = await import("../api/forecast.js");

const servidor = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PUERTO}`);

  if (url.pathname === "/api/forecast") {
    // adaptador mínimo: añade res.status()/res.json() al objeto http nativo
    res.status = (c) => (res.statusCode = c, res);
    res.json = (o) => (res.setHeader("Content-Type", "application/json"), res.end(JSON.stringify(o)));
    // en local siempre datos frescos: ignora el Cache-Control de CDN del handler
    const _setHeader = res.setHeader.bind(res);
    res.setHeader = (k, v) => _setHeader(k, /^cache-control$/i.test(k) ? "no-store" : v);
    res.setHeader("Cache-Control", "no-store");
    return forecastHandler(req, res);
  }

  let ruta = url.pathname === "/" ? "/index.html" : url.pathname;
  ruta = normalize(ruta).replace(/^(\.\.[/\\])+/, "");
  const archivo = resolve(raizPublica, "." + ruta);
  if (!archivo.startsWith(raizPublica)) {
    res.statusCode = 403;
    return res.end("Prohibido");
  }
  try {
    const cuerpo = await readFile(archivo);
    res.setHeader("Content-Type", TIPOS[extname(archivo)] || "application/octet-stream");
    res.end(cuerpo);
  } catch {
    res.statusCode = 404;
    res.end("No encontrado");
  }
});

servidor.listen(PUERTO, () => {
  console.log(`Dashboard en  http://localhost:${PUERTO}`);
  console.log(`API en        http://localhost:${PUERTO}/api/forecast`);
});
