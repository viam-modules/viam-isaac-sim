/// <reference types="svelte" />
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_VIAM_HOST?: string;
  readonly VITE_VIAM_API_KEY?: string;
  readonly VITE_VIAM_API_KEY_ID?: string;
  readonly VITE_RESOURCE_NAME?: string;
}
