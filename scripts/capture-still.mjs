// Full-page screenshot of the demo after the race has finished (4 s at speed=256).
// Same dependency-free approach as capture-frames.mjs.
// usage: node scripts/capture-still.mjs <url> <out.png> [width=1240] [height=1000]
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const [url, out, w = "1240", h = "1000"] = process.argv.slice(2);
const port = 9800 + Math.floor(Math.random() * 100);
const chrome = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`, "--user-data-dir=/tmp/oa-chrome-" + port, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); } }
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send("Emulation.setDeviceMetricsOverride", { width: Number(w), height: Number(h), deviceScaleFactor: 1, mobile: false });
await send("Page.enable");
await send("Page.navigate", { url });
await sleep(4000);
const r = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: Number(w), height: Number(h), scale: 1 } });
writeFileSync(out, Buffer.from(r.result.data, "base64"));
console.log(out); ws.close(); chrome.kill(); process.exit(0);
