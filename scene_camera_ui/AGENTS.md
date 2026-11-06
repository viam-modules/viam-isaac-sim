# scene_camera-control

Guidance for any coding agent working in this project. Read it before changing `src/`.

## What this is

This project is one custom control for one Viam model. It is a Svelte 5 app that builds to a
single self-contained file, `dist/index.html`. The Viam app loads that file in a sandboxed iframe
on a machine's control page, next to the resource that uses this model. A module with several
models has one project like this per model, each in its own directory.

The control never talks to the machine directly. Every API call goes over `window.postMessage` to
the Viam app, which holds the machine connection and relays calls through
`@viamrobotics/post-message-sdk`.

## Stack and commands

| Piece   | Choice                                                                    |
| ------- | ------------------------------------------------------------------------- |
| UI      | Svelte 5 with runes, `@viamrobotics/prime-core` components                |
| Styling | Tailwind v4 with `@viamrobotics/tailwind-config` tokens, in `src/app.css` |
| Bridge  | `@viamrobotics/post-message-sdk/svelte`                                   |
| Build   | Vite with `vite-plugin-singlefile`, one `index.html` in, one file out     |
| Tests   | Vitest with `@testing-library/svelte` in jsdom                            |

| Command               | What it does                                                       |
| --------------------- | ------------------------------------------------------------------ |
| `pnpm dev`            | Serves `dev/host.html`, a stand-in for the Viam app, with HMR      |
| `pnpm build`          | Builds `dist/index.html`                                           |
| `pnpm check:artifact` | Fails unless the artifact is one self-contained file under 1 MiB   |
| `pnpm check`          | `svelte-check` type errors                                         |
| `pnpm lint`           | ESLint                                                             |
| `pnpm test`           | Vitest                                                             |
| `pnpm verify`         | Format, lint, check, test, build, and the artifact check, in order |

Before saying a change is done, run `pnpm verify` and read its output.

## Talking to the machine

Use the hooks from `@viamrobotics/post-message-sdk/svelte`, and nothing else, to reach the
machine. They have the same call shapes as `@viamrobotics/svelte-sdk`.

- `useBridge({ reportSize: true })` runs once in `src/main.ts`. It starts the handshake with the
  host and exposes `status`, `resourceName`, and `error` reactively.
- `createResourceClient<SensorClient>()` returns a handle for the resource this control was
  rendered for. The host sends that resource's name in the handshake, so the control needs no
  configuration. The handle is `undefined` until the bridge is connected.
- `createResourceQuery(client, 'getReadings', { refetchInterval: 1000 })` polls a method and
  exposes `data`, `error`, `isLoading`, `isFetching`, `isError`, `isSuccess`, and `refetch`.
- `createResourceMutation(client, 'doCommand')` exposes `mutate`, `mutateAsync`, `isPending`, and
  `reset`. `mutate` takes the method's arguments as one tuple: `mutate([{ command: 'status' }])`.
- `createResourceStream(client, 'getVideo')` collects a server stream into `data`.

Import `@viamrobotics/sdk` with `import type` only. The client class is a type parameter, as in
`createResourceClient<SensorClient>()`. A value import of a client class pulls the whole SDK
runtime into the artifact and can push it over the size limit.

Errors from the bridge are `BridgeError` instances with a `kind`: `not-connected`, `timeout`,
`disconnected`, `not-allowed`, or a remote error relayed from the machine. Show `error.message`
to the user and branch on `kind` only when the control can do something different about it.

## What the sandbox forbids

The Viam app injects a Content Security Policy into the artifact, and `index.html` carries the
same policy so failures show up in `pnpm dev`. The policy closes everything except inline script
and style. Do not write any of these, because they fail at runtime and the artifact check or the
browser will reject them:

- `fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, or any other network request
- `import()` of a remote URL, external `<script>`, `<link>`, or `<img src="https://…">`
- `<form>` submission or `<base>`. The sandbox blocks a submission before the `submit` event
  fires, so an `onsubmit` handler never runs and `preventDefault` cannot help. Use a
  `<button type="button">` with `onclick`, and handle the Enter key on inputs with `onkeydown`.
- Reading the machine config, other resources, or the app's session. The host allows calls to the
  owning resource only.

Everything the control needs from the machine comes through the hooks above. Everything else
(images, fonts, icons) is bundled into the artifact or omitted.

## Artifact rules

- `pnpm build` must emit exactly one file, `dist/index.html`, with every script, style, font, and
  image inlined. `pnpm check:artifact` enforces this and a 1 kB to 1 MiB size window.
- No `http://` or `https://` URL may appear in the artifact, except the identifier-only prefixes
  `http://www.w3.org/` and `https://svelte.dev/e/`.
- `@viamrobotics/tailwind-config/fonts` is not imported, because it adds about 300 kB. The control
  uses the fallback font stack. Keep it that way unless the artifact stays well under the limit.
- `dev/` is the local stand-in host and is not part of the build.

## Conventions

- Svelte 5 runes only: `$state`, `$derived`, `$effect`, `$props`. No `export let`, no legacy
  stores, no `onMount` for data loading when a hook covers it.
- Prefer `@viamrobotics/prime-core` components (`Button`, `Badge`, inputs) and tailwind-config
  tokens over hand-rolled elements and raw colors, so the control looks like the rest of the app.
- Show the four states every control has: connecting (`bridge.status`), loading, error, and
  data. An empty data set is a state too.
- Once data has arrived, keep it on screen. The app polls every second and the host reconnects
  now and then, so `isFetching`, a `connecting` bridge, or a failed poll changes an indicator or
  a notice, never the whole view. Placeholders render only before the first data.
- Tests mock `@viamrobotics/post-message-sdk/svelte` with `vi.mock` and assert the rendered DOM.
  See `src/App.test.ts`.
- `src/App.svelte` is the control. Keep it small and move logic into `src/*.svelte.ts` modules
  when it grows.

## Local development

`pnpm dev` opens `dev/host.html`, which dials the machine named in `.env` (`VITE_VIAM_HOST`,
`VITE_VIAM_API_KEY`, `VITE_VIAM_API_KEY_ID`) and renders the control in a sandboxed iframe. With
those unset it serves a fake robot, so the UI works offline. `VITE_RESOURCE_NAME` is the resource
the control may address in dev. The dev server widens the policy only as far as Vite's own scripts
and HMR socket need.

## Shipping

The module's `meta.json` declares the built file on the model entry, for example
`"custom_control": "station_ui/dist/index.html"`, and the module build packages it. The
module tarball must contain `meta.json` at its root, because the registry reads the entry from
that copy. A tarball without it ships a module whose control the app cannot find. The full
flow, including `viam module reload`, is in the deploy guide this project was scaffolded with
(`docs/deploy.md` in `viamrobotics/viam-post-message-sdk`).

## Review checklist

When asked to review a change here, check each line and name the one that fails:

1. Machine access goes through the `/svelte` hooks only. No `fetch` or other network call.
2. `@viamrobotics/sdk` is imported as types only.
3. No external URL, script, stylesheet, font, or image was added.
4. The connecting, loading, error, and empty states still render, and data stays mounted while
   `isFetching` is true, the bridge is `connecting`, or a later poll fails.
5. `pnpm verify` passes, and `check:artifact` reports one file under 1 MiB.

## If you add a `CLAUDE.md`

Claude Code reads this file on its own only while no `CLAUDE.md` exists in or above this project.
If you add one, put `@AGENTS.md` on a line of its own inside it so this guidance keeps loading.
