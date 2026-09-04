import { readFile } from "node:fs/promises";

async function load() {
  const url = new URL("../data/forecast.json", import.meta.url);
  return JSON.parse(await readFile(url, "utf8"));
}

export default async function handler(req, res) {
  try {
    const data = await load();
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader(
      "Cache-Control",
      "public, max-age=300, s-maxage=900, stale-while-revalidate=3600"
    );
    res.status(200).end(JSON.stringify({ ...data, servido: new Date().toISOString() }));
  } catch (err) {
    res.status(500).json({ error: "No se pudo cargar el pronóstico", detalle: String(err) });
  }
}
