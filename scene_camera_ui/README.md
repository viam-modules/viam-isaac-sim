# scene_camera-control

A custom control for one Viam model: a Svelte 5 + Tailwind v4 app that builds to a single
self-contained `dist/index.html`. The Viam app loads that file in a sandboxed iframe and relays
the control's machine API calls through `@viamrobotics/post-message-sdk`.

The control shows the live image of a `viam-labs:isaac-sim:scene_camera` (polled about every
125 ms, ~8 fps) and drives its view. It polls `doCommand {command: 'get_view'}` every 2 s and sends
`set_view` patches (`target`, `distance`, `yaw_deg`, `pitch_deg`, `fov_deg`) and `reset_view`.

- Drag to orbit, shift-drag or two-finger drag to pan, wheel or pinch to zoom.
- Double-click, the Home button, or the H/Home keys return to the home view.
- Arrow keys orbit 5 degrees, and +/- zoom.
- The FOV slider sends on release.
- The view locks while an episode records.

It reaches the machine only through `@viamrobotics/post-message-sdk/svelte`.

## Develop

```sh
cp .env.example .env   # then fill it in
pnpm install
pnpm dev
```

`pnpm dev` opens `/dev/host.html`, a stand-in for the Viam app. It dials the machine named by
`VITE_VIAM_HOST`, `VITE_VIAM_API_KEY` and `VITE_VIAM_API_KEY_ID` and hosts `App.svelte` in a
sandboxed iframe, with HMR. With those unset it serves a fake robot, so the UI works offline.
`VITE_RESOURCE_NAME` (default `scene-camera`) is the resource the control may address. Add
`?recording` to the dev host URL to see the recording lock.

The dev server widens the CSP only as far as Vite needs (`'self'` scripts and styles, a WebSocket
for HMR). Network access to the internet is blocked in dev as it is in production. `dev/` is not part
of the build.

## Build

```sh
pnpm build
pnpm verify
```

`pnpm build` emits exactly one file, `dist/index.html`, with all JS, CSS, fonts and images inlined.
`pnpm check:artifact` fails unless `dist/` holds only that file, it contains no `http(s)://` URL
(except the identifier-only prefixes `http://www.w3.org/` and `https://svelte.dev/e/`, which never cause a request), it carries the CSP meta tag from `index.html`, and its size
is between 1 kB and 1 MiB.

Fonts: `@viamrobotics/tailwind-config/fonts` is not imported. Built with it the artifact is about
714 kB. Without it, the artifact is about 414 kB. Our limit for keeping the fonts is 512 kB, so they stay out and
text uses the fallback font stack. To use them, add this line to `src/main.ts` and keep the
artifact under the 1 MiB check.

```ts
import '@viamrobotics/tailwind-config/fonts';
```

`pnpm verify` runs format, lint, check, test, build and check:artifact in order.

## Deploy

The root `meta.json` already references the built file on the model entry:

```json
"custom_control": "scene_camera_ui/dist/index.html"
```

The root `make module.tar.gz` builds the control and packages it. See also
[the deploy guide](https://github.com/viamrobotics/viam-post-message-sdk/blob/main/docs/deploy.md).

## What the CSP blocks

The Viam app injects this policy into every artifact, and `index.html` carries it verbatim:

```
default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'
```

Inline scripts and styles run. Everything else is closed: no `fetch`, XHR, WebSocket or EventSource
(`connect-src 'none'`), no external scripts, stylesheets, fonts or images (only `data:` and `blob:`
for images, `blob:` for media, `data:` for fonts), no `<base>` and no form submission. The only way
out is the postMessage bridge.
