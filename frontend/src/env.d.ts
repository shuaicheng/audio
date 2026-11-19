/// <reference types="vite/client" />

declare module "*.js?raw" {
  const source: string
  export default source
}

interface ImportMetaEnv {
  readonly VITE_WS_URL?: string
  readonly VITE_APP_AUTH_TOKEN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
