import {
  DEFAULT_HYPERPARAMS,
  SCHEDULE_NAMES,
  SCHEDULE_RULES,
  optimizerIds,
  race,
  scheduleIds,
  surfaces,
  surfaceById,
  type Hyperparams,
  type OptimizerId,
  type RaceEntry,
  type ScheduleId,
  type Surface,
  type Vec2,
} from "../src/index.ts";
import {
  HEIGHT,
  buildMesh,
  clampCamera,
  lambert,
  painterOrder,
  project,
  quadNormal,
  toWorldXY,
  viewMatrix,
  type Camera,
  type Mesh,
  type Vec3,
} from "../src/view3d.ts";

const COLORS: Record<string, string> = {
  sgd: "#e6e8ee",
  momentum: "#4cc38a",
  nesterov: "#3ea8ff",
  rmsprop: "#c678dd",
  adam: "#f5a524",
  adagrad: "#e06c75",
  adamw: "#d19a66",
  lion: "#56d4c8",
};
const NAMES: Record<string, string> = {
  sgd: "Gradient descent",
  momentum: "Momentum",
  nesterov: "Nesterov",
  rmsprop: "RMSProp",
  adam: "Adam",
  adagrad: "AdaGrad",
  adamw: "AdamW",
  lion: "Lion",
};
const RULES: Record<string, string> = {
  sgd: "θ ← θ − η∇f(θ)",
  momentum: "v ← βv − η∇f(θ);  θ ← θ + v",
  nesterov: "v ← βv − η∇f(θ + βv);  θ ← θ + v",
  rmsprop: "s ← ρs + (1−ρ)g²;  θ ← θ − ηg/(√s + ε)",
  adam: "m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  θ ← θ − η·m̂/(√v̂ + ε)",
  adagrad: "G ← G + g²;  θ ← θ − ηg/(√G + ε)",
  adamw: "Adam's m̂, v̂;  θ ← θ − η(m̂/(√v̂ + ε) + λθ)",
  lion: "c = β₁m + (1−β₁)g;  θ ← θ − η·sign(c);  m ← β₂m + (1−β₂)g",
};
const FORMULAS: Record<string, string> = {
  quadratic: "f = ½(x² + 10y²)",
  rosenbrock: "f = (1 − x)² + 100(y − x²)²",
  beale: "f = (1.5 − x + xy)² + (2.25 − x + xy²)² + (2.625 − x + xy³)²",
  saddle: "f = x² − y²",
  himmelblau: "f = (x² + y − 11)² + (x + y² − 7)²",
  rastrigin: "f = 20 + Σ (xᵢ² − 10 cos 2πxᵢ)",
  ackley: "f = −20 exp(−0.2 √(½(x² + y²))) − exp(½(cos 2πx + cos 2πy)) + e + 20",
  booth: "f = (x + 2y − 7)² + (2x + y − 5)²",
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("plot");
const ctx = canvas.getContext("2d")!;
const surfaceSelect = $<HTMLSelectElement>("surface");
const speedSelect = $<HTMLSelectElement>("speed");
const legendBody = document.querySelector<HTMLTableSectionElement>("#legend tbody")!;
const chart = $<HTMLCanvasElement>("chart");
const cctx = chart.getContext("2d")!;

// ---- hyperparameter sliders -------------------------------------------------
// η, β₂ and ε span several orders of magnitude, so those sliders move in log space:
// η = 10^u, ε = 10^u, and β₂ = 1 − 10^u (so the slider reaches 0.9999 smoothly).
const tune = $<HTMLInputElement>("tune");
const sliders = {
  lr: { el: $<HTMLInputElement>("lr"), to: (u: number) => 10 ** u, from: (v: number) => Math.log10(v) },
  beta: { el: $<HTMLInputElement>("beta"), to: (u: number) => u, from: (v: number) => v },
  rho: { el: $<HTMLInputElement>("rho"), to: (u: number) => u, from: (v: number) => v },
  beta1: { el: $<HTMLInputElement>("beta1"), to: (u: number) => u, from: (v: number) => v },
  beta2: { el: $<HTMLInputElement>("beta2"), to: (u: number) => 1 - 10 ** u, from: (v: number) => Math.log10(1 - v) },
  eps: { el: $<HTMLInputElement>("eps"), to: (u: number) => 10 ** u, from: (v: number) => Math.log10(v) },
  weightDecay: { el: $<HTMLInputElement>("weightDecay"), to: (u: number) => 10 ** u, from: (v: number) => Math.log10(v) },
  lionBeta1: { el: $<HTMLInputElement>("lionBeta1"), to: (u: number) => u, from: (v: number) => v },
  lionBeta2: { el: $<HTMLInputElement>("lionBeta2"), to: (u: number) => u, from: (v: number) => v },
};
type SliderKey = keyof typeof sliders;
const DEFAULT_LR = 0.01;

const value = (k: SliderKey) => sliders[k].to(Number(sliders[k].el.value));
const fmt = (k: SliderKey, v: number): string => {
  if (k === "eps" || k === "lr" || k === "weightDecay") return v >= 0.01 ? String(+v.toPrecision(2)) : v.toExponential(1);
  if (k === "beta2" || k === "lionBeta2") return String(+v.toFixed(4));
  return String(+v.toFixed(3));
};
function readHyperparams(): Hyperparams {
  return {
    beta: value("beta"),
    rho: value("rho"),
    beta1: value("beta1"),
    beta2: value("beta2"),
    eps: value("eps"),
    weightDecay: value("weightDecay"),
    lionBeta1: value("lionBeta1"),
    lionBeta2: value("lionBeta2"),
  };
}
function showSliderValues(): void {
  for (const k of Object.keys(sliders) as SliderKey[]) $(`${k}-out`).textContent = fmt(k, value(k));
  sliders.lr.el.disabled = tune.checked;
  $("lr-out").textContent = tune.checked ? "tuned" : fmt("lr", value("lr"));
}
function resetSliders(): void {
  const defaults: Record<SliderKey, number> = { lr: DEFAULT_LR, ...DEFAULT_HYPERPARAMS };
  for (const k of Object.keys(sliders) as SliderKey[]) sliders[k].el.value = String(sliders[k].from(defaults[k]));
  tune.checked = true;
}

for (const s of surfaces) surfaceSelect.add(new Option(s.name, s.id));

// ---- learning-rate schedule and which optimizers race ---------------------------
const scheduleSelect = $<HTMLSelectElement>("schedule");
for (const id of scheduleIds) scheduleSelect.add(new Option(SCHEDULE_NAMES[id], id));
const racers = $<HTMLFieldSetElement>("racers");
const racerBoxes = new Map<OptimizerId, HTMLInputElement>();
for (const id of optimizerIds) {
  const label = document.createElement("label");
  label.className = "check";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.value = id;
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = COLORS[id]!;
  label.append(box, swatch, ` ${NAMES[id]}`);
  racers.append(label);
  racerBoxes.set(id, box);
}
const selectedRacers = (): OptimizerId[] => optimizerIds.filter((id) => racerBoxes.get(id)!.checked);

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

// ---- 3D surface view ----------------------------------------------------------
// The math (rotations, perspective, height map, mesh, painter's order) lives in
// src/view3d.ts; this part only turns it into canvas paths.
type View = "2d" | "3d";
let view: View = "2d";
const MESH_N = 56;
const LIGHT: Vec3 = (() => {
  const l: Vec3 = [-0.45, -0.55, 0.7];
  const n = Math.hypot(...l);
  return [l[0] / n, l[1] / n, l[2] / n];
})();
const DEFAULT_CAMERA: Camera = { yaw: -0.6, pitch: -1.0, distance: 4.4, focal: 920, cx: W / 2, cy: H / 2 + 10 };
let camera: Camera = { ...DEFAULT_CAMERA };
let mesh: Mesh | null = null;
/** Per-quad base colour (from height) and brightness (from the fixed light), cached per surface. */
let quadShade: { rgb: Vec3; light: number }[] = [];

function buildSurface3d(): void {
  mesh = buildMesh(surface, MESH_N);
  const m = mesh;
  quadShade = m.quads.map((q) => {
    const [a, b, c, d] = q.map((i) => m.vertices[i]!) as [Vec3, Vec3, Vec3, Vec3];
    const t = (a[2] + b[2] + c[2] + d[2]) / (4 * HEIGHT);
    // Same blue → violet → warm ramp as the contour map, brightened for shading.
    const rgb: Vec3 = [30 + 170 * t, 40 + 90 * t, 90 + 150 * (1 - t) * t * 2 + 40 * t];
    return { rgb, light: lambert(quadNormal(a, b, c, d), LIGHT) };
  });
}

/** World position of a trajectory point: on the floor grid at (x, y), lifted to its loss height. */
function worldPoint(p: Vec2, f: number, h: (f: number) => number): Vec3 {
  const [wx, wy] = toWorldXY(surface.bounds, p[0], p[1]);
  return [Math.min(1, Math.max(-1, wx)), Math.min(1, Math.max(-1, wy)), h(f) * HEIGHT + 0.012];
}

function draw3d(): void {
  if (!mesh) buildSurface3d();
  const m = mesh!;
  const R = viewMatrix(camera);
  const proj = m.vertices.map((v) => project(v, camera, R));

  // Everything that can hide something else goes into one list, sorted far → near.
  type Item = { depth: number; paint: () => void };
  const items: Item[] = [];
  m.quads.forEach((q, k) => {
    const [a, b, c, d] = q.map((i) => proj[i]!) as [typeof proj[0], typeof proj[0], typeof proj[0], typeof proj[0]];
    const { rgb, light } = quadShade[k]!;
    const col = `rgb(${rgb[0] * light | 0},${rgb[1] * light | 0},${rgb[2] * light | 0})`;
    const edge = `rgb(${rgb[0] * light * 0.78 | 0},${rgb[1] * light * 0.78 | 0},${rgb[2] * light * 0.78 | 0})`;
    items.push({
      depth: (a.depth + b.depth + c.depth + d.depth) / 4,
      paint: () => {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.lineTo(c.x, c.y);
        ctx.lineTo(d.x, d.y);
        ctx.closePath();
        ctx.fillStyle = col;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      },
    });
  });

  // Trajectory segments join the same sort, so a ridge in front can hide a path behind it.
  // Long jumps are split into short pieces so each piece sorts against the quads it crosses,
  // and every piece gets a small bias towards the camera so it wins against the quad under it.
  const cell = 2 / MESH_N;
  const heads: { p: Vec3; color: string }[] = [];
  for (const e of entries) {
    const t = e.trajectory;
    if (!t) continue;
    const n = Math.min(t.points.length, frame + 1);
    const color = COLORS[e.optimizer]!;
    const stride = Math.max(1, Math.floor(n / 600));
    let prev = worldPoint(t.points[0]!, t.losses[0]!, m.height);
    for (let k = stride; k < n + stride - 1; k += stride) {
      const idx = Math.min(k, n - 1);
      const cur = worldPoint(t.points[idx]!, t.losses[idx]!, m.height);
      const len = Math.hypot(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]);
      const pieces = Math.min(12, Math.max(1, Math.ceil(len / (1.5 * cell))));
      for (let s = 0; s < pieces; s++) {
        const lerp = (u: number): Vec3 => [
          prev[0] + (cur[0] - prev[0]) * u,
          prev[1] + (cur[1] - prev[1]) * u,
          prev[2] + (cur[2] - prev[2]) * u,
        ];
        const p0 = project(lerp(s / pieces), camera, R);
        const p1 = project(lerp((s + 1) / pieces), camera, R);
        items.push({
          depth: (p0.depth + p1.depth) / 2 - 0.03,
          paint: () => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.2;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          },
        });
      }
      prev = cur;
    }
    if (n > 0) heads.push({ p: worldPoint(t.points[n - 1]!, t.losses[n - 1]!, m.height), color });
  }

  ctx.fillStyle = "#171a21";
  ctx.fillRect(0, 0, W, H);
  for (const i of painterOrder(items.map((it) => it.depth))) items[i]!.paint();

  // Markers sit on top of everything: minima (white ×), the start (ring), and each optimizer's head.
  for (const mn of surface.minima) {
    const q = project(worldPoint(mn, surface.f(mn[0], mn[1]), m.height), camera, R);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(q.x - 5, q.y - 5);
    ctx.lineTo(q.x + 5, q.y + 5);
    ctx.moveTo(q.x + 5, q.y - 5);
    ctx.lineTo(q.x - 5, q.y + 5);
    ctx.stroke();
  }
  const s0 = project(worldPoint(start, surface.f(start[0], start[1]), m.height), camera, R);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(s0.x, s0.y, 7, 0, 2 * Math.PI);
  ctx.stroke();
  for (const { p, color } of heads) {
    const q = project(p, camera, R);
    ctx.fillStyle = color;
    ctx.strokeStyle = "#0f1115";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(q.x, q.y, 5.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  ctx.font = "15px ui-monospace, Menlo, monospace";
  ctx.fillStyle = "#8b93a7";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(m.mode === "log" ? "height: log(1 + f − f_lo), log-scaled" : "height: f, linear", 12, 12);
  ctx.textAlign = "right";
  ctx.fillText("drag: orbit · wheel: zoom · double-click: reset", W - 12, H - 26);
}

function render(): void {
  if (view === "3d") draw3d();
  else draw();
}

function setView(v: View): void {
  view = v;
  $("view-2d").setAttribute("aria-pressed", String(v === "2d"));
  $("view-3d").setAttribute("aria-pressed", String(v === "3d"));
  canvas.classList.toggle("orbit", v === "3d");
  render();
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

/**
 * Loss against step for every optimizer, drawn up to the current frame. With a
 * finite minimum the y-axis is log₁₀(f − f*), which turns linear convergence
 * into a straight line; on the saddle (no minimum) it is f itself, linear.
 */
function drawChart(longest: number): void {
  const CW = chart.width;
  const CH = chart.height;
  const pad = { l: 64, r: 36, t: 14, b: 34 };
  const pw = CW - pad.l - pad.r;
  const ph = CH - pad.t - pad.b;
  const logScale = Number.isFinite(surface.minValue);
  const FLOOR = 1e-12;
  const yOf = (f: number) => (logScale ? Math.log10(Math.max(f - surface.minValue, FLOOR)) : f);

  let lo = Infinity;
  let hi = -Infinity;
  for (const e of entries) for (const f of e.trajectory?.losses ?? []) {
    const y = yOf(f);
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  if (!Number.isFinite(lo)) return;
  if (logScale) {
    lo = Math.floor(lo);
    hi = Math.ceil(hi);
  } else {
    lo = Math.max(lo, -2);
  }
  if (hi - lo < 1e-9) hi = lo + 1;
  const xs = Math.max(1, longest);
  const px = (k: number) => pad.l + (k / xs) * pw;
  const py = (y: number) => pad.t + (1 - (Math.min(Math.max(y, lo), hi) - lo) / (hi - lo)) * ph;

  cctx.clearRect(0, 0, CW, CH);
  cctx.font = "20px ui-monospace, Menlo, monospace";
  cctx.fillStyle = "#8b93a7";
  cctx.strokeStyle = "#262b36";
  cctx.lineWidth = 1;
  // horizontal grid: one line per decade on the log scale, five otherwise
  const yTicks = logScale
    ? Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).filter((_, i, a) => a.length <= 14 || i % 2 === 0)
    : Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  cctx.textAlign = "right";
  cctx.textBaseline = "middle";
  for (const y of yTicks) {
    cctx.beginPath();
    cctx.moveTo(pad.l, py(y));
    cctx.lineTo(CW - pad.r, py(y));
    cctx.stroke();
    cctx.fillText(logScale ? `1e${y}` : y.toFixed(2), pad.l - 8, py(y));
  }
  cctx.textAlign = "center";
  cctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) {
    const k = Math.round((xs * i) / 5);
    cctx.fillText(String(k), px(k), CH - pad.b + 8);
  }

  for (const e of entries) {
    const losses = e.trajectory?.losses ?? [];
    const n = Math.min(losses.length, frame + 1);
    if (n === 0) continue;
    cctx.strokeStyle = COLORS[e.optimizer]!;
    cctx.lineWidth = 2.5;
    cctx.beginPath();
    // Thin long runs to at most ~2 points per pixel column.
    const stride = Math.max(1, Math.floor(n / (2 * pw)));
    for (let k = 0; k < n; k += stride) {
      if (k === 0) cctx.moveTo(px(k), py(yOf(losses[k]!)));
      else cctx.lineTo(px(k), py(yOf(losses[k]!)));
    }
    cctx.lineTo(px(n - 1), py(yOf(losses[n - 1]!)));
    cctx.stroke();
  }
  $("chart-axis").textContent = logScale
    ? "· y: f − f* on a log scale (floored at 1e-12) · x: step"
    : "· y: f (no minimum, linear) · x: step";
}

function tick(): void {
  const longest = Math.max(0, ...entries.map((e) => (e.trajectory?.points.length ?? 1) - 1));
  frame = Math.min(longest, frame + Number(speedSelect.value));
  render();
  updateLegend();
  drawChart(longest);
  if (frame < longest) raf = requestAnimationFrame(tick);
}

function restart(): void {
  cancelAnimationFrame(raf);
  const saddle = !Number.isFinite(surface.minValue);
  const lrGrid = saddle ? [0.01] : tune.checked ? undefined : [value("lr")];
  const schedule = scheduleSelect.value as ScheduleId;
  entries = race(surface, {
    steps: 3000,
    start,
    lrGrid,
    hyperparams: readHyperparams(),
    schedule,
    optimizers: selectedRacers(),
  });
  showSliderValues();
  frame = 0;
  $("formula").textContent = `${FORMULAS[surface.id] ?? ""}   ·   ${SCHEDULE_RULES[schedule]}`;
  $("note").textContent = saddle
    ? "The saddle has no minimum, so every optimizer uses η = 0.01 and the race ends when f < −1. Start exactly on the x-axis and nobody escapes."
    : tune.checked
      ? "Each optimizer uses its best learning rate from {1e-4 … 1} for this start point; a ✓ means f − f* < 1e-6."
      : `Every optimizer uses η = ${fmt("lr", value("lr"))}; "diverged" means it blew up at that η. A ✓ means f − f* < 1e-6.`;
  raf = requestAnimationFrame(tick);
}

function selectSurface(id: string): void {
  surface = surfaceById(id);
  start = surface.start;
  background = renderBackground();
  mesh = null;
  restart();
}

surfaceSelect.addEventListener("change", () => selectSurface(surfaceSelect.value));
$("replay").addEventListener("click", restart);

// Rerun the race live while a slider is dragged, at most once per animation frame.
let pending = 0;
const rerun = () => {
  showSliderValues();
  cancelAnimationFrame(pending);
  pending = requestAnimationFrame(restart);
};
for (const { el } of Object.values(sliders)) el.addEventListener("input", rerun);
tune.addEventListener("change", rerun);
scheduleSelect.addEventListener("change", rerun);
racers.addEventListener("change", rerun);
$("reset").addEventListener("click", () => {
  resetSliders();
  rerun();
});
$("view-2d").addEventListener("click", () => setView("2d"));
$("view-3d").addEventListener("click", () => setView("3d"));

// 3D: drag to orbit (horizontal = yaw, vertical = pitch), wheel to zoom, double-click to reset.
let drag: { x: number; y: number } | null = null;
canvas.addEventListener("pointerdown", (ev) => {
  if (view !== "3d") return;
  drag = { x: ev.clientX, y: ev.clientY };
  canvas.setPointerCapture(ev.pointerId);
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  camera = clampCamera({
    ...camera,
    yaw: camera.yaw + (ev.clientX - drag.x) * 0.01,
    pitch: camera.pitch + (ev.clientY - drag.y) * 0.008,
  });
  drag = { x: ev.clientX, y: ev.clientY };
  render();
});
const endDrag = () => {
  drag = null;
  canvas.classList.remove("dragging");
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener(
  "wheel",
  (ev) => {
    if (view !== "3d") return;
    ev.preventDefault();
    camera = clampCamera({ ...camera, distance: camera.distance * Math.exp(ev.deltaY * 0.001) });
    render();
  },
  { passive: false },
);
canvas.addEventListener("dblclick", () => {
  if (view !== "3d") return;
  camera = { ...DEFAULT_CAMERA };
  render();
});

canvas.addEventListener("click", (ev) => {
  if (view !== "2d") return;
  const r = canvas.getBoundingClientRect();
  start = fromPx(((ev.clientX - r.left) / r.width) * W, ((ev.clientY - r.top) / r.height) * H);
  restart();
});

// ?surface=rosenbrock&speed=256&view=3d&schedule=cosine&only=adam,adamw,lion opens a specific
// setup (handy for links and screenshots).
const query = new URLSearchParams(location.search);
const sched = query.get("schedule");
if (sched && (scheduleIds as string[]).includes(sched)) scheduleSelect.value = sched;
const only = query.get("only")?.split(",");
if (only?.some((id) => racerBoxes.has(id as OptimizerId))) {
  for (const [id, box] of racerBoxes) box.checked = only.includes(id);
}
const requested = query.get("surface");
const speed = query.get("speed");
if (speed && [...speedSelect.options].some((o) => o.value === speed)) speedSelect.value = speed;
if (requested && surfaces.some((s) => s.id === requested)) surfaceSelect.value = requested;
resetSliders();
selectSurface(surfaceSelect.value);
if (query.get("view") === "3d") setView("3d");
