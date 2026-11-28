// Grabs the demo's canvas as PNG frames while the race animates, for the README GIFs.
// No dependencies: drives headless Chrome over the DevTools protocol with Node's built-in
// fetch and WebSocket (Node 22+). Frames that repeat are skipped; it stops once the
// canvas has not changed for six polls (the race has finished).
// usage: node scripts/capture-frames.mjs <url> <outdir> [intervalMs=80] [maxFrames=60] [selector=#plot]
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const [url, out, interval = "80", maxFrames = "60", sel = "#plot"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", `--remote-debugging-port=${port}`, "--window-size=1400,1800", "--user-data-dir=/tmp/oa-chrome-" + port, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); } }
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send("Page.enable");
await send("Page.navigate", { url });
await sleep(300);
let last = "", n = 0, same = 0;
for (let f = 0; f < Number(maxFrames); f++) {
  const r = await send("Runtime.evaluate", { expression: `document.querySelector(${JSON.stringify(sel)}).toDataURL("image/png")`, returnByValue: true });
  const data = r.result.result.value;
  if (data === last) { if (++same > 5) break; } else { same = 0; last = data; writeFileSync(`${out}/f${String(n++).padStart(3, "0")}.png`, Buffer.from(data.split(",")[1], "base64")); }
  await sleep(Number(interval));
}
console.log(out, n, "frames");
ws.close(); chrome.kill();
process.exit(0);
