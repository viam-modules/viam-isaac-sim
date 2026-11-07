import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const DIST = 'dist';
const MIN_BYTES = 1024; // 1 kB
const MAX_BYTES = 1024 * 1024; // 1 MiB
const BYTES_PER_KB = 1000;
const CSP =
  "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'";

const fail = (message) => {
  console.error(`check:artifact failed: ${message}`);
  process.exit(1);
};

let files;
try {
  files = readdirSync(DIST, { recursive: true }).filter((f) => statSync(join(DIST, f)).isFile());
} catch {
  fail(`${DIST}/ does not exist; run the build first.`);
}
if (files.length !== 1 || files[0] !== 'index.html') {
  fail(`${DIST}/ must hold exactly one file, index.html; found [${files.join(', ')}].`);
}

const html = readFileSync(join(DIST, 'index.html'), 'utf8');
const size = Buffer.byteLength(html);

// Identifier-only prefixes that never cause a request.
const urls = (html.match(/https?:\/\/(?!www\.w3\.org\/|svelte\.dev\/e\/)/g) ?? []).length;
if (urls !== 0)
  fail(`found ${String(urls)} http(s):// URL(s); the artifact must be self-contained.`);

const cspMeta = /<meta\s+http-equiv="Content-Security-Policy"\s+content="([^"]*)"/.exec(html);
if (cspMeta?.[1] !== CSP) {
  fail('the Content-Security-Policy meta tag is missing or altered.');
}
if (size < MIN_BYTES) fail(`size ${String(size)} B is under the ${String(MIN_BYTES)} B minimum.`);
if (size > MAX_BYTES) fail(`size ${String(size)} B is over the ${String(MAX_BYTES)} B maximum.`);

console.log(`check:artifact ok: dist/index.html is ${(size / BYTES_PER_KB).toFixed(1)} kB`);
