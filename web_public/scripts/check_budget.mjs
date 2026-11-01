// Fail if any built page exceeds the weight budget or loads a third-party asset.
// Weight is HTML plus the CSS and JS it references, uncompressed, which is stricter
// than what travels over the wire.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const DIST = "dist";
const BUDGET_BYTES = 80 * 1024;
const ASSET = /<(?:link[^>]+href|script[^>]+src)="([^"]+)"/g;

function htmlFiles(directory) {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      return htmlFiles(path);
    }
    return name.endsWith(".html") ? [path] : [];
  });
}

const failures = [];
const pages = htmlFiles(DIST);
if (pages.length === 0) {
  failures.push("no pages found: run the build first");
}

for (const page of pages) {
  const html = readFileSync(page, "utf8");
  let total = Buffer.byteLength(html);
  for (const [, url] of html.matchAll(ASSET)) {
    if (/^(https?:)?\/\//.test(url)) {
      if (!url.startsWith("https://notitiacivica.com/")) {
        failures.push(`${page}: third-party asset ${url}`);
      }
      continue;
    }
    if (/\.(css|js)$/.test(url)) {
      total += statSync(join(DIST, url)).size;
    }
  }
  const kilobytes = (total / 1024).toFixed(1);
  console.log(`${kilobytes.padStart(6)} KB  ${relative(DIST, page)}`);
  if (total >= BUDGET_BYTES) {
    failures.push(`${page}: ${kilobytes} KB is over the 80 KB budget`);
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
