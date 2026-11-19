export interface TtsFrame {
  turnId: string
  flags: number
  segmentIndex: number
  chunkIndex: number
  sampleRate: number
  channels: number
  pcm: Int16Array
}

export function encodeUpstream(seq: number, sampleRate: number, pcm: Int16Array): ArrayBuffer {
  const bytes = new Uint8Array(16 + pcm.byteLength)
  const view = new DataView(bytes.buffer)
  bytes.set([0x41, 0x55, 0x50, 0x43], 0)
  view.setUint8(4, 1)
  view.setUint8(5, 0)
  view.setUint32(6, seq, true)
  view.setUint32(10, sampleRate, true)
  view.setUint16(14, 1, true)
  bytes.set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength), 16)
  return bytes.buffer
}

export function decodeDownstream(buffer: ArrayBuffer): TtsFrame | null {
  if (buffer.byteLength < 19) return null
  const view = new DataView(buffer)
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))
  if (magic !== "ADNA" || view.getUint8(4) !== 1) return null
  const flags = view.getUint8(5)
  const idLength = view.getUint8(6)
  const idStart = 7
  if (buffer.byteLength < idStart + idLength + 12) return null
  const turnId = new TextDecoder().decode(new Uint8Array(buffer, idStart, idLength))
  let offset = idStart + idLength
  const segmentIndex = view.getUint16(offset, true)
  offset += 2
  const chunkIndex = view.getUint32(offset, true)
  offset += 4
  const sampleRate = view.getUint32(offset, true)
  offset += 4
  const channels = view.getUint16(offset, true)
  offset += 2
  const payload = new Uint8Array(buffer, offset)
  const frameBytes = channels * 2
  const usable = frameBytes > 0 ? payload.byteLength - (payload.byteLength % frameBytes) : 0
  const copy = new Uint8Array(usable)
  copy.set(payload.subarray(0, usable))
  return {
    turnId,
    flags,
    segmentIndex,
    chunkIndex,
    sampleRate,
    channels,
    pcm: new Int16Array(copy.buffer),
  }
}

export function defaultWsUrl(): string {
  const configured = import.meta.env.VITE_WS_URL
  if (configured) return configured
  const protocol = location.protocol === "https:" ? "wss:" : "ws:"
  return `${protocol}//${location.host}/api/ws`
}
