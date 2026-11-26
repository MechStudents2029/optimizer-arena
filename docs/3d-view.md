# The 3D view, frame by frame

The README's [3D section](../README.md#the-3d-view-how-a-surface-gets-onto-a-flat-canvas)
explains the math. This page follows one frame through the code, so the pieces in
`src/view3d.ts` and `web/main.ts` can be read in the order they run.

## Once per surface

`buildMesh(surface, 56)` samples $f$ on a $57 \times 57$ grid over the plot
window (3249 evaluations), picks the height mode with `chooseHeightMode`, and
returns:

- `vertices`: $(x_w, y_w, z)$ with $x_w, y_w \in [-1, 1]$ and $z \in [0, 0.9]$,
- `quads`: four vertex indices each, row by row in $y$,
- `height`: the $f \mapsto [0, 1]$ map, kept so trajectories use exactly the
  same scale as the mesh.

The demo then caches, per quad, a base colour from its mean height and a
brightness from `lambert(quadNormal(...), LIGHT)`. The light is fixed to the
world, not the camera, so when you orbit, the lit side of a hill stays lit,
which is what makes the shape readable while it turns.

Nothing here depends on the camera, so dragging never rebuilds the mesh; it is
rebuilt only when the surface changes.

## Every frame

1. **One matrix.** `viewMatrix(camera)` computes $R = R_x(\phi) R_z(\psi)$ once.
   Every point is then a single $3 \times 3$ multiply instead of two.
2. **Project the vertices.** `project(v, camera, R)` returns screen $(x, y)$ and
   a depth for each of the 3249 vertices. Quads share vertices, so projecting
   vertices (not quad corners) does each one once instead of up to four times.
3. **Collect drawables.** Each quad becomes an item with the mean depth of its
   corners. Each trajectory step from $\theta_k$ to $\theta_{k+1}$ is a 3D chord
   between $(\theta_k, h(f_k))$ and $(\theta_{k+1}, h(f_{k+1}))$, lifted 0.012
   above the surface and cut into pieces about 1.5 grid cells long (at most 12),
   each an item with its own depth minus a 0.03 bias.
4. **Sort and paint.** `painterOrder(depths)` returns indices far to near and the
   items paint in that order: fill plus a slightly darker stroke per quad (the
   stroke hides the hairline seams that anti-aliasing leaves between filled
   neighbours, and doubles as the wireframe), a 2.2 px round-capped line per path
   piece.
5. **Markers last.** Minima (×), the start point (ring) and each optimizer's
   current position (dot) are drawn on top of everything, so a racer is never
   lost behind a hill.

## Why cut the paths into pieces

Painter's algorithm sorts whole items, so an item can only be entirely in front
of or entirely behind another. A momentum step on Rosenbrock can cross a third of
the plot window in one jump; as a single segment it would sort by its midpoint and
be drawn over hills it actually passes behind (or vanish under quads it is in
front of). Pieces about one cell long sort against the quads they actually cross.

The bias works the other way: a piece lying on the surface has almost the same
depth as the quad beneath it, and without the bias half of them would lose that
tie and be painted over. 0.03 world units is a little under one cell (2/56 ≈ 0.036), enough to win
against the quad underneath and too small to beat a ridge that is really in front.

## Cost

Measured with Node 24 on the box this was written on (per call, mean of many):

| step | time |
| --- | --- |
| `buildMesh(rosenbrock, 56)` (once per surface) | 1.3 ms |
| project 3249 vertices + depth-sort 3136 quads | 0.7 ms |
| sort 6000 items (mesh + a few thousand path pieces) | 1.4 ms |

So the math is a few milliseconds a frame; the canvas fills and strokes are the
real cost, which is why long trajectories are thinned in the 3D view to every
$\lfloor n/600 \rfloor$-th point, at most about 1200 drawn steps per optimizer (the 2D view still draws every step).

## Things it deliberately does not do

- **No z-buffer.** Per-pixel depth would fix the rare misordering of two long,
  steep quads, but needs WebGL or a software rasteriser; neither is worth it for
  a smooth surface at this resolution.
- **No back-face culling.** From above the floor (the pitch is clamped to
  $(-\pi/2, 0]$) almost every quad faces the camera, and the steep sides of
  Rosenbrock's walls that do face away are almost always covered by nearer quads.
- **No clipping.** The zoom is clamped (distance 2.5 to 12 world units) so the
  camera never enters the scene and no point ever reaches depth zero; `project`
  still guards the divide, and a test checks it.
