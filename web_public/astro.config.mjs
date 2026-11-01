import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://notitiacivica.com",
  output: "static",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "never" },
  devToolbar: { enabled: false },
  telemetry: false,
});
