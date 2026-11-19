export function alignedFrameCount(byteLength: number, channels: number): number {
  const frameBytes = Math.max(1, channels) * 2
  if (byteLength <= 0) return 0
  return Math.floor(byteLength / frameBytes)
}

export function bufferDuration(frames: number, sampleRate: number): number {
  if (frames <= 0 || sampleRate <= 0) return 0
  return frames / sampleRate
}

export function nextStartTime(currentTime: number, scheduledEnd: number, lead = 0.03): number {
  return Math.max(currentTime + lead, scheduledEnd)
}

export class PcmPlayer {
  private context: AudioContext | null = null
  private gain: GainNode | null = null
  private sources: AudioBufferSourceNode[] = []
  private scheduledEnd = 0
  private activeTurn = ""
  private droppedTurn = ""
  private volume = 1
  private muted = false
  private holding = false

  constructor(
    private readonly maxAheadSec = 1.8,
    private readonly onHold: (hold: boolean) => void = () => undefined,
    private readonly onPlayed: (bytes: number) => void = () => undefined,
  ) {}

  async resume(): Promise<void> {
    const ctx = this.ensure()
    if (ctx.state !== "running") await ctx.resume()
  }

  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume))
    this.applyGain()
  }

  setMuted(muted: boolean): void {
    this.muted = muted
    this.applyGain()
  }

  enqueue(pcm: Int16Array, sampleRate: number, channels: number, turnId: string): number {
    if (!turnId || turnId === this.droppedTurn) return 0
    if (this.activeTurn && this.activeTurn !== turnId) this.stop(this.activeTurn)
    this.activeTurn = turnId
    const frames = Math.floor(pcm.length / Math.max(1, channels))
    if (frames <= 0 || sampleRate <= 0) return 0
    const ctx = this.ensure()
    const buffer = ctx.createBuffer(channels, frames, sampleRate)
    for (let channel = 0; channel < channels; channel += 1) {
      const data = buffer.getChannelData(channel)
      for (let index = 0; index < frames; index += 1) {
        data[index] = (pcm[index * channels + channel] ?? 0) / 32768
      }
    }
    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(this.gain as GainNode)
    const startAt = nextStartTime(ctx.currentTime, this.scheduledEnd)
    source.start(startAt)
    this.scheduledEnd = startAt + buffer.duration
    this.sources.push(source)
    const bytes = frames * channels * 2
    source.onended = () => {
      this.sources = this.sources.filter((item) => item !== source)
      this.onPlayed(bytes)
      this.updateHold()
    }
    this.updateHold()
    return bytes
  }

  stop(turnId?: string): void {
    if (turnId) this.droppedTurn = turnId
    for (const source of this.sources) {
      try {
        source.onended = null
        source.stop()
      } catch {
        /* already stopped */
      }
    }
    this.sources = []
    this.scheduledEnd = 0
    this.activeTurn = ""
    if (this.holding) {
      this.holding = false
      this.onHold(false)
    }
  }

  private ensure(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext()
      this.gain = this.context.createGain()
      this.gain.connect(this.context.destination)
      this.applyGain()
    }
    return this.context
  }

  private applyGain(): void {
    if (!this.gain) return
    this.gain.gain.value = this.muted ? 0 : this.volume
  }

  private updateHold(): void {
    if (!this.context) return
    const ahead = Math.max(0, this.scheduledEnd - this.context.currentTime)
    const next = ahead > this.maxAheadSec
    if (next !== this.holding) {
      this.holding = next
      this.onHold(next)
    }
  }
}
