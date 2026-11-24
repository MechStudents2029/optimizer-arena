import { race, surfaces, surfaceById, type RaceEntry, type Surface, type Vec2 } from "../src/index.ts";

const COLORS: Record<string, string> = {
  sgd: "#e6e8ee",
  momentum: "#4cc38a",
  nesterov: "#3ea8ff",
  rmsprop: "#c678dd",
  adam: "#f5a524",
};
const NAMES: Record<string, string> = {
  sgd: "Gradient descent",
  momentum: "Momentum",
  nesterov: "Nesterov",
  rmsprop: "RMSProp",
  adam: "Adam",
};
const RULES: Record<string, string> = {
  sgd: "θ ← θ − η∇f(θ)",
  momentum: "v ← βv − η∇f(θ);  θ ← θ + v",
  nesterov: "v ← βv − η∇f(θ + βv);  θ ← θ + v",
  rmsprop: "s ← ρs + (1−ρ)g²;  θ ← θ − ηg/(√s + ε)",
  adam: "m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  θ ← θ − η·m̂/(√v̂ + ε)",
};
const FORMULAS: Record<string, string> = {
  quadratic: "f = ½(x² + 10y²)",
  rosenbrock: "f = (1 − x)² + 100(y − x²)²",
  beale: "f = (1.5 − x + xy)² + (2.25 − x + xy²)² + (2.625 − x + xy³)²",
  saddle: "f = x² − y²",
  himmelblau: "f = (x² + y − 11)² + (x + y² − 7)²",
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("plot");
const ctx = canvas.getContext("2d")!;
const surfaceSelect = $<HTMLSelectElement>("surface");
const speedSelect = $<HTMLSelectElement>("speed");
const legendBody = document.querySelector<HTMLTableSectionElement>("#legend tbody")!;

for (const s of surfaces) surfaceSelect.add(new Option(s.name, s.id));

let surface: Surface = surfaces[0]!;
let start: Vec2 = surface.start;
let entries: RaceEntry[] = [];
let background: ImageData | null = null;
let frame = 0;
let raf = 0;

const W = canvas.width;
const H = canvas.height;
const toPx = ([x, y]: Vec2): Vec2 => {
  const [x0, x1, y0, y1] = surface.bounds;
  return [((x - x0) / (x1 - x0)) * W, (1 - (y - y0) / (y1 - y0)) * H];
};
const fromPx = (px: number, py: number): Vec2 => {
  const [x0, x1, y0, y1] = surface.bounds;
  return [x0 + (px / W) * (x1 - x0), y1 - (py / H) * (y1 - y0)];
};

/** Heatmap of log(1 + f − f_lo) with banding every 1/4 unit, so bands read as contours. */
function renderBackground(): ImageData {
  const img = ctx.createImageData(W, H);
  const vals = new Float64Array(W * H);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const [x, y] = fromPx(i + 0.5, j + 0.5);
      const v = surface.f(x, y);
      vals[j * W + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const top = Math.log1p(hi - lo);
  for (let k = 0; k < vals.length; k++) {
    const t = Math.log1p(vals[k]! - lo) / top; // 0 (low) … 1 (high)
    const band = Math.floor(t * 24) / 24;
    const edge = (t * 24) % 1 < 0.06 ? 0.75 : 1; // thin darker line at each band edge
    img.data[4 * k] = (18 + 70 * band) * edge;
    img.data[4 * k + 1] = (24 + 40 * band) * edge;
    img.data[4 * k + 2] = (48 + 90 * (1 - band) * band * 2 + 40 * band) * edge;
    img.data[4 * k + 3] = 255;
  }
  return img;
}

function draw(): void {
  ctx.putImageData(background!, 0, 0);
  for (const m of surface.minima) {
    const [px, py] = toPx(m);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px - 6, py - 6);
    ctx.lineTo(px + 6, py + 6);
    ctx.moveTo(px + 6, py - 6);
    ctx.lineTo(px - 6, py + 6);
    ctx.stroke();
  }
  for (const e of entries) {
    const pts = e.trajectory?.points ?? [];
    const n = Math.min(pts.length, frame + 1);
    ctx.strokeStyle = COLORS[e.optimizer]!;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const [px, py] = toPx(pts[k]!);
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (n > 0) {
      const [px, py] = toPx(pts[n - 1]!);
      ctx.fillStyle = COLORS[e.optimizer]!;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  const [sx, sy] = toPx(start);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(sx, sy, 7, 0, 2 * Math.PI);
  ctx.stroke();
}

function updateLegend(): void {
  legendBody.innerHTML = "";
  for (const e of entries) {
    const t = e.trajectory;
    const shown = t ? Math.min(frame, t.points.length - 1) : 0;
    const loss = t ? t.losses[shown]! : NaN;
    const done = t && shown === t.points.length - 1;
    const status = e.lr === null ? "diverged" : done && e.steps !== null ? `${e.steps} ✓` : String(shown);
    const row = legendBody.insertRow();
    row.innerHTML =
      `<td><span class="swatch" style="background:${COLORS[e.optimizer]}"></span></td>` +
      `<td>${NAMES[e.optimizer]}</td>` +
      `<td class="num">${e.lr ?? "—"}</td>` +
      `<td class="num">${status}</td>` +
      `<td class="num">${Number.isFinite(loss) ? loss.toExponential(2) : "—"}</td>` +
      `<td class="rule">${RULES[e.optimizer]}</td>`;
  }
}

function tick(): void {
  const longest = Math.max(0, ...entries.map((e) => (e.trajectory?.points.length ?? 1) - 1));
  frame = Math.min(longest, frame + Number(speedSelect.value));
  draw();
  updateLegend();
  if (frame < longest) raf = requestAnimationFrame(tick);
}

function restart(): void {
  cancelAnimationFrame(raf);
  const saddle = !Number.isFinite(surface.minValue);
  entries = race(surface, { steps: 3000, start, lrGrid: saddle ? [0.01] : undefined });
  frame = 0;
  $("formula").textContent = FORMULAS[surface.id] ?? "";
  $("note").textContent = saddle
    ? "The saddle has no minimum, so every optimizer uses η = 0.01 and the race ends when f < −1. Start exactly on the x-axis and nobody escapes."
    : "Each optimizer uses its best learning rate from {1e-4 … 1} for this start point; a ✓ means f − f* < 1e-6.";
  raf = requestAnimationFrame(tick);
}

function selectSurface(id: string): void {
  surface = surfaceById(id);
  start = surface.start;
  background = renderBackground();
  restart();
}

surfaceSelect.addEventListener("change", () => selectSurface(surfaceSelect.value));
$("replay").addEventListener("click", restart);
canvas.addEventListener("click", (ev) => {
  const r = canvas.getBoundingClientRect();
  start = fromPx(((ev.clientX - r.left) / r.width) * W, ((ev.clientY - r.top) / r.height) * H);
  restart();
});

// ?surface=rosenbrock&speed=256 opens a specific surface (handy for links and screenshots).
const query = new URLSearchParams(location.search);
const requested = query.get("surface");
const speed = query.get("speed");
if (speed && [...speedSelect.options].some((o) => o.value === speed)) speedSelect.value = speed;
if (requested && surfaces.some((s) => s.id === requested)) surfaceSelect.value = requested;
selectSurface(surfaceSelect.value);
