#!/usr/bin/env bash
# Regenerates docs/screenshots from the built demo: two stills, two GIFs, one 3D still.
# Needs google-chrome on PATH and ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."
npm run build >/dev/null
npx vite preview --port 4173 --strictPort >/dev/null 2>&1 &
preview=$!
trap 'kill $preview' EXIT
sleep 2
base=http://localhost:4173
out=docs/screenshots
tmp=$(mktemp -d)

node scripts/capture-still.mjs "$base/?surface=rosenbrock&speed=256" "$tmp/2d.png" 1240 1110
node scripts/capture-still.mjs "$base/?surface=rosenbrock&speed=256&view=3d" "$tmp/3d.png" 1240 1110
ffmpeg -y -loglevel error -i "$tmp/2d.png" -vf crop=1240:1088:0:0 "$out/rosenbrock.png"
ffmpeg -y -loglevel error -i "$tmp/3d.png" -vf crop=1240:1088:0:0 "$out/rosenbrock-3d.png"

for view in 2d 3d; do
  q="surface=rosenbrock&speed=16"; [ "$view" = 3d ] && q="$q&view=3d"
  node scripts/capture-frames.mjs "$base/?$q" "$tmp/f$view" 60 200
  # 12 fps, hold the last frame 2 s, 480 px wide, 128-colour palette.
  ffmpeg -y -loglevel error -framerate 12 -i "$tmp/f$view/f%03d.png" \
    -vf "tpad=stop_mode=clone:stop_duration=2,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
    -loop 0 "$out/rosenbrock-$view.gif"
done

node scripts/capture-frames.mjs "$base/?surface=himmelblau&speed=256&view=3d" "$tmp/h3d" 100 60
cp "$(ls "$tmp"/h3d/*.png | tail -1)" "$out/himmelblau-3d.png"
rm -rf "$tmp"
