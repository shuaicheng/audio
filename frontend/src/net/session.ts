import { decodeDownstream, defaultWsUrl, type TtsFrame } from "./protocol"

export type ServerEvent = Record<string, unknown> & { type?: string }

export class VoiceSocket {
  private socket: WebSocket | null = null
  private url: string

  constructor(
    private readonly onEvent: (event: ServerEvent) => void,
    private readonly onAudio: (frame: TtsFrame) => void,
    private readonly onStatus: (status: "connecting" | "open" | "closed") => void,
  ) {
    this.url = defaultWsUrl()
  }

  connect(): void {
    this.close()
    this.onStatus("connecting")
    const socket = new WebSocket(this.url)
    socket.binaryType = "arraybuffer"
    this.socket = socket
    socket.onopen = () => {
      const token = import.meta.env.VITE_APP_AUTH_TOKEN
      if (token) socket.send(JSON.stringify({ type: "auth", token }))
      this.onStatus("open")
    }
    socket.onclose = () => {
      if (this.socket === socket) this.onStatus("closed")
    }
    socket.onerror = () => {
      this.onEvent({ type: "error", message: "和服务器的连接中断。" })
    }
    socket.onmessage = (message) => {
      if (typeof message.data === "string") {
        try {
          this.onEvent(JSON.parse(message.data) as ServerEvent)
        } catch {
          this.onEvent({ type: "error", message: "收到无法解析的消息。" })
        }
        return
      }
      const frame = decodeDownstream(message.data as ArrayBuffer)
      if (frame && frame.pcm.length) this.onAudio(frame)
    }
  }

  send(payload: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(payload))
  }

  sendAudio(frame: ArrayBuffer): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(frame)
  }

  close(): void {
    this.socket?.close()
    this.socket = null
  }
}
