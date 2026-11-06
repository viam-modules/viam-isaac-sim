# scene_camera-control

A custom control for one Viam model: a Svelte 5 + Tailwind v4 app that builds to a single
self-contained `dist/index.html`. The Viam app loads that file in a sandboxed iframe and relays
the control's machine API calls through `@viamrobotics/post-message-sdk`.

`src/App.svelte` shows the readings of the sensor named by `VITE_RESOURCE_NAME`, polled every
second, with `@viamrobotics/prime-core` components and Tailwind styling. It reaches the machine only
through `@viamrobotics/post-message-sdk/svelte`.

## Develop

```sh
cp .env.example .env   # then fill it in
pnpm install
pnpm dev
```

`pnpm dev` opens `/dev/host.html`, a stand-in for the Viam app. It dials the machine named by
`VITE_VIAM_HOST`, `VITE_VIAM_API_KEY` and `VITE_VIAM_API_KEY_ID` and hosts `App.svelte` in a
sandboxed iframe, with HMR. With those unset it serves a fake robot, so the UI works offline.
`VITE_RESOURCE_NAME` (default `my-sensor`) is the resource the control may address.

The dev server widens the CSP only as far as Vite needs (`'self'` scripts and styles, a WebSocket
for HMR). Network access to the internet is blocked in dev as it is in production. `dev/` is not part
of the build.

## Build

```sh
pnpm build
pnpm check:artifact
```

`pnpm build` emits exactly one file, `dist/index.html`, with all JS, CSS, fonts and images inlined.
`pnpm check:artifact` fails unless `dist/` holds only that file, it contains no `http(s)://` URL
(except the identifier-only prefixes `http://www.w3.org/` and `https://svelte.dev/e/`, which never cause a request), it carries the CSP meta tag from `index.html`, and its size
is between 1 kB and 1 MiB.

Fonts: `@viamrobotics/tailwind-config/fonts` is not imported. Built with it the artifact is about
714 kB. Without it, the artifact is about 397 kB. Our limit for keeping the fonts is 512 kB, so they stay out and
text uses the fallback font stack. To use them, add this line to `src/main.ts` and keep the
artifact under the 1 MiB check.

```ts
import '@viamrobotics/tailwind-config/fonts';
```

`pnpm verify` runs format, lint, check, test, build and check:artifact in order.

## Deploy

Reference the built file from the model entry in your module's `meta.json`:

```json
{
  "api": "rdk:component:sensor",
  "model": "my-org:my-module:my-sensor",
  "custom_control": "my_sensor_ui/dist/index.html"
}
```

Your resource must also serve this file, and your module build must package it. Both steps are in
[the deploy guide](../../docs/deploy.md).

## What the CSP blocks

The Viam app injects this policy into every artifact, and `index.html` carries it verbatim:

```
default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'
```

Inline scripts and styles run. Everything else is closed: no `fetch`, XHR, WebSocket or EventSource
(`connect-src 'none'`), no external scripts, stylesheets, fonts or images (only `data:` and `blob:`
for images, `blob:` for media, `data:` for fonts), no `<base>` and no form submission. The only way
out is the postMessage bridge.
