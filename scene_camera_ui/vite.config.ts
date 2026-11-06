import { resolve } from 'node:path';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

/** Widens the artifact's CSP just enough for Vite's dev server: its scripts, styles and HMR socket. */
const devCsp = (): Plugin => ({
  name: 'dev-csp',
  apply: 'serve',
  transformIndexHtml: (html) =>
    html
      .replace('script-src ', "script-src 'self' ")
      .replace('style-src ', "style-src 'self' ")
      .replace("connect-src 'none'", "connect-src 'self' ws: wss:"),
});

// Larger than any asset the control could carry, so every one is inlined into the single file.
const INLINE_EVERY_ASSET_BYTES = 100_000_000;

export default defineConfig({
  plugins: [svelte(), tailwindcss(), viteSingleFile(), devCsp()],
  build: {
    cssCodeSplit: false,
    assetsInlineLimit: INLINE_EVERY_ASSET_BYTES,
    rollupOptions: { input: resolve(import.meta.dirname, 'index.html') },
  },
});
