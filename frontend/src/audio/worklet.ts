import workletSource from "./pcm-capture-worklet.js?raw"

/** 独立脚本字符串，避免生产压缩把函数改名后，worklet 里仍调用 downmix。 */
export function buildWorkletSource(): string {
  return workletSource
}

export function workletModuleUrl(): string {
  const blob = new Blob([workletSource], { type: "application/javascript" })
  return URL.createObjectURL(blob)
}
