import { defineConfig } from "vite";

// The demo page lives in web/ and imports the library straight from src/.
export default defineConfig({
  root: "web",
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
