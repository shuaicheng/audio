/** 在识别通道就绪前先攒住音频帧，就绪后按原顺序送出。停止后不再接收迟到帧。 */
export class AudioSendGate {
  private open = false
  private closed = false
  private queue: ArrayBuffer[] = []

  constructor(private readonly send: (frame: ArrayBuffer) => void) {}

  get isOpen(): boolean {
    return this.open
  }

  push(frame: ArrayBuffer): void {
    if (this.closed || frame.byteLength === 0) return
    if (!this.open) {
      this.queue.push(frame)
      return
    }
    this.send(frame)
  }

  release(): void {
    if (this.closed || this.open) return
    this.open = true
    const queued = this.queue
    this.queue = []
    for (const frame of queued) this.send(frame)
  }

  close(): ArrayBuffer[] {
    this.closed = true
    if (this.open) return []
    const leftover = this.queue
    this.queue = []
    return leftover
  }
}
