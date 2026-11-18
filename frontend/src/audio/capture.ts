import { workletModuleUrl } from "./worklet"
import { encodeUpstream } from "../net/protocol"

const TARGET_RATE = 16000
const FLUSH_SAMPLES = 1600

export interface CaptureStats {
  frames: number
  samples: number
  level: number
  contextState: string
}

export function microphoneBlockReason(isSecureContext: boolean, hasGetUserMedia: boolean): string | null {
  if (!isSecureContext) {
    return "当前页面不是安全上下文。本机 localhost 可以录音，远程访问必须使用 HTTPS，否则浏览器不会打开麦克风。"
  }
  if (!hasGetUserMedia) {
    return "这个浏览器没有提供麦克风接口。请换用新版浏览器，远程访问时还要使用 HTTPS。"
  }
  return null
}

export function pcmLevel(pcm: Int16Array): number {
  if (pcm.length === 0) return 0
  let sum = 0
  for (let index = 0; index < pcm.length; index += 1) {
    const sample = (pcm[index] ?? 0) / 32768
    sum += sample * sample
  }
  return Math.min(1, Math.sqrt(sum / pcm.length))
}

export async function resumeOrThrow(context: { state: string; resume: () => Promise<void> }): Promise<void> {
  if (context.state === "suspended") {
    try {
      await context.resume()
    } catch {
      throw new DOMException("浏览器没有允许启动录音。", "InvalidStateError")
    }
  }
  if (context.state !== "running") {
    throw new DOMException("录音上下文仍被挂起，请再点一次「开始说话」。", "InvalidStateError")
  }
}

export function micErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      return "浏览器拒绝了麦克风权限。请在地址栏允许麦克风；远程打开页面时必须使用 HTTPS。"
    }
    if (error.name === "NotFoundError") return "没有找到可用的麦克风。"
    if (error.name === "SecurityError" || error.name === "NotSupportedError") {
      return error.message || "当前页面不能使用麦克风。localhost 可以用，远程访问必须走 HTTPS / WSS。"
    }
    if (error.name === "InvalidStateError") {
      return error.message || "录音上下文被挂起，请再点一次「开始说话」。"
    }
    if (error.message) return error.message
  }
  return "无法打开麦克风。"
}

export async function startCapture(hooks: {
  onFrame: (frame: ArrayBuffer) => void
  onStats?: (stats: CaptureStats) => void
  onError?: (message: string) => void
}): Promise<() => Promise<void>> {
  const blocked = microphoneBlockReason(window.isSecureContext, !!navigator.mediaDevices?.getUserMedia)
  if (blocked) throw new DOMException(blocked, "SecurityError")
  const context = new AudioContext()
  await resumeOrThrow(context)
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    video: false,
  })
  await resumeOrThrow(context)
  const url = workletModuleUrl()
  try {
    await context.audioWorklet.addModule(url)
  } catch (error) {
    URL.revokeObjectURL(url)
    stream.getTracks().forEach((track) => track.stop())
    await context.close()
    const detail = error instanceof Error ? error.message : ""
    throw new DOMException(detail ? `录音模块没有加载：${detail}` : "录音模块没有加载。", "InvalidStateError")
  }
  URL.revokeObjectURL(url)
  const source = context.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(context, "pcm-capture")
  node.onprocessorerror = () => {
    hooks.onError?.("录音处理模块出错，浏览器没有交出采样。")
  }
  const silent = context.createGain()
  silent.gain.value = 0
  let pending = new Int16Array(0)
  let seq = 1
  let frames = 0
  let samples = 0
  let closed = false
  const publish = (pcm: Int16Array) => {
    samples += pcm.length
    hooks.onStats?.({
      frames,
      samples,
      level: pcmLevel(pcm),
      contextState: context.state,
    })
  }
  const flush = (force: boolean) => {
    if (closed || pending.length === 0) return
    if (!force && pending.length < FLUSH_SAMPLES) return
    const frame = encodeUpstream(seq, TARGET_RATE, pending)
    seq += 1
    frames += 1
    pending = new Int16Array(0)
    hooks.onStats?.({ frames, samples, level: 0, contextState: context.state })
    hooks.onFrame(frame)
  }
  node.port.onmessage = (event: MessageEvent<ArrayBuffer | { type?: string; message?: string; sampleRate?: number }>) => {
    if (closed) return
    const payload = event.data
    if (!(payload instanceof ArrayBuffer)) {
      if (payload && payload.type === "error") hooks.onError?.(payload.message || "录音处理模块出错。")
      if (payload && payload.type === "ready") {
        const rate = "sampleRate" in payload && typeof payload.sampleRate === "number" ? payload.sampleRate : 0
        hooks.onStats?.({ frames, samples, level: 0, contextState: rate ? `running@${rate}` : "running" })
      }
      return
    }
    const chunk = new Int16Array(payload)
    publish(chunk)
    const merged = new Int16Array(pending.length + chunk.length)
    merged.set(pending, 0)
    merged.set(chunk, pending.length)
    pending = merged
    flush(false)
  }
  source.connect(node)
  node.connect(silent)
  silent.connect(context.destination)
  await resumeOrThrow(context)
  hooks.onStats?.({ frames, samples, level: 0, contextState: context.state })
  return async () => {
    closed = true
    flush(true)
    node.port.onmessage = null
    node.disconnect()
    source.disconnect()
    silent.disconnect()
    stream.getTracks().forEach((track) => track.stop())
    await context.close()
  }
}
