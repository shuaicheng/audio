import { workletModuleUrl } from "./worklet"
import { encodeUpstream } from "../net/protocol"

const TARGET_RATE = 16000
const FLUSH_SAMPLES = 1600

export function micErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : ""
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "浏览器拒绝了麦克风权限。远程打开页面时必须使用 HTTPS。"
  }
  if (name === "NotFoundError") return "没有找到可用的麦克风。"
  if (name === "SecurityError" || name === "NotSupportedError") {
    return "当前页面不能使用麦克风。localhost 可以用，远程访问必须走 HTTPS / WSS。"
  }
  return "无法打开麦克风。"
}

export async function startCapture(send: (frame: ArrayBuffer) => void): Promise<() => Promise<void>> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new DOMException("麦克风不可用", "SecurityError")
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    video: false,
  })
  const context = new AudioContext()
  const url = workletModuleUrl()
  await context.audioWorklet.addModule(url)
  URL.revokeObjectURL(url)
  const source = context.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(context, "pcm-capture")
  const silent = context.createGain()
  silent.gain.value = 0
  let pending = new Int16Array(0)
  let seq = 1
  const flush = (force: boolean) => {
    if (pending.length === 0) return
    if (!force && pending.length < FLUSH_SAMPLES) return
    const frame = encodeUpstream(seq, TARGET_RATE, pending)
    seq += 1
    pending = new Int16Array(0)
    send(frame)
  }
  node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
    const chunk = new Int16Array(event.data)
    const merged = new Int16Array(pending.length + chunk.length)
    merged.set(pending, 0)
    merged.set(chunk, pending.length)
    pending = merged
    flush(false)
  }
  source.connect(node)
  node.connect(silent)
  silent.connect(context.destination)
  return async () => {
    flush(true)
    node.port.onmessage = null
    node.disconnect()
    source.disconnect()
    stream.getTracks().forEach((track) => track.stop())
    await context.close()
  }
}
