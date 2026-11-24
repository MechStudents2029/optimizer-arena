import { defineConfig } from "vitest/config";

// Tests run from the repo root (vite.config.ts points Vite at web/ for the demo).
export default defineConfig({
  test: { include: ["test/**/*.test.ts"] },
});
