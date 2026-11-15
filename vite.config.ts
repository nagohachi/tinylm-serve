import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base so the build works under any GitHub Pages path (/<repo>/).
export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: { format: "es" },
  build: { target: "es2022" },
});
