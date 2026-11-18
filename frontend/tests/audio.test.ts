import { describe, expect, it } from "vitest"
import { formatLevel, microphoneBlockReason, pcmLevel, resumeOrThrow } from "../src/audio/capture"
import { AudioSendGate } from "../src/audio/sendGate"
import { alignedFrameCount, bufferDuration, nextStartTime } from "../src/audio/playback"
import { createResampleState, downmix, floatToInt16, resampleBlock } from "../src/audio/resample"
import { buildWorkletSource } from "../src/audio/worklet"
import { decodeDownstream, encodeUpstream } from "../src/net/protocol"

describe("resample", () => {
  it("converts a constant 48 kHz signal to 16 kHz pcm16", () => {
    const input = new Float32Array(48000)
    input.fill(0.25)
    const state = createResampleState()
    const output = resampleBlock(input, 48000, 16000, state)
    expect(output.length).toBeGreaterThan(15800)
    expect(output.length).toBeLessThanOrEqual(16000)
    expect(output[100]).toBeGreaterThan(7000)
    expect(output[100]).toBeLessThan(9000)
  })

  it("keeps the same rate as real pcm16 conversion", () => {
    const input = new Float32Array([0, 0.5, -1])
    const output = resampleBlock(input, 16000, 16000, createResampleState())
    expect(Array.from(output)).toEqual([0, 16384, -32767])
  })

  it("downmixes stereo before resampling", () => {
    const mono = downmix([new Float32Array([0, 1]), new Float32Array([1, 1])])
    expect(Array.from(mono)).toEqual([0.5, 1])
    expect(floatToInt16(new Float32Array([1]))[0]).toBe(32767)
  })

  it("ships a self-contained worklet whose calls match its declarations", () => {
    const source = buildWorkletSource()
    expect(source).toContain('registerProcessor("pcm-capture"')
    for (const name of ["downmix", "resampleBlock", "floatToInt16"]) {
      expect(source).toContain(`function ${name}`)
      expect(source).toContain(`${name}(`)
    }
    expect(source).not.toContain("${")
  })
})

describe("playback alignment", () => {
  it("aligns start time and drops partial frames", () => {
    expect(nextStartTime(10, 9.5)).toBeCloseTo(10.03)
    expect(nextStartTime(10, 12)).toBe(12)
    expect(alignedFrameCount(5, 1)).toBe(2)
    expect(bufferDuration(24000, 24000)).toBe(1)
    expect(bufferDuration(16000, 24000)).toBeCloseTo(16000 / 24000)
  })
})

describe("microphone startup", () => {
  it("explains insecure context and suspended capture", async () => {
    expect(microphoneBlockReason(false, true)).toContain("HTTPS")
    expect(microphoneBlockReason(true, false)).toContain("麦克风")
    expect(microphoneBlockReason(true, true)).toBeNull()
    const context = { state: "suspended", async resume() { this.state = "running" } }
    await resumeOrThrow(context)
    expect(context.state).toBe("running")
    await expect(resumeOrThrow({ state: "suspended", async resume() {} })).rejects.toThrow(/挂起/)
  })

  it("buffers frames until recognition is ready and then keeps order", () => {
    const sent: string[] = []
    const gate = new AudioSendGate((frame) => sent.push(new TextDecoder().decode(frame)))
    gate.push(new TextEncoder().encode("a").buffer)
    gate.push(new TextEncoder().encode("b").buffer)
    expect(sent).toEqual([])
    gate.release()
    gate.push(new TextEncoder().encode("c").buffer)
    expect(sent).toEqual(["a", "b", "c"])
    expect(gate.close()).toEqual([])
    gate.push(new TextEncoder().encode("d").buffer)
    expect(sent).toEqual(["a", "b", "c"])
  })

  it("returns unsent frames when recording stops before ready", () => {
    const gate = new AudioSendGate(() => undefined)
    gate.push(new TextEncoder().encode("tail").buffer)
    const leftover = gate.close()
    expect(new TextDecoder().decode(leftover[0])).toBe("tail")
    expect(pcmLevel(new Int16Array([0, 0]))).toBe(0)
    expect(pcmLevel(new Int16Array([32767]))).toBeGreaterThan(0.99)
    expect(formatLevel(0.004)).toBe("0.4")
    expect(formatLevel(0.15)).toBe("15")
  })
})

describe("binary frames", () => {
  it("roundtrips upstream audio and downstream metadata", () => {
    const pcm = new Int16Array([1, -2, 3])
    const encoded = encodeUpstream(4, 16000, pcm)
    const bytes = new Uint8Array(encoded)
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe("AUPC")
    const view = new DataView(encoded)
    expect(view.getUint32(6, true)).toBe(4)
    expect(view.getUint32(10, true)).toBe(16000)
    const header = new Uint8Array([
      0x41, 0x44, 0x4e, 0x41, 1, 1, 8,
      ..."turn_abc".split("").map((char) => char.charCodeAt(0)),
    ])
    const rest = new DataView(new ArrayBuffer(12))
    rest.setUint16(0, 3, true)
    rest.setUint32(2, 9, true)
    rest.setUint32(6, 22050, true)
    rest.setUint16(10, 1, true)
    const payload = new Uint8Array(pcm.buffer)
    const frame = new Uint8Array(header.length + 12 + payload.byteLength)
    frame.set(header, 0)
    frame.set(new Uint8Array(rest.buffer), header.length)
    frame.set(payload, header.length + 12)
    const decoded = decodeDownstream(frame.buffer)
    expect(decoded?.turnId).toBe("turn_abc")
    expect(decoded?.sampleRate).toBe(22050)
    expect(decoded?.segmentIndex).toBe(3)
    expect(decoded?.chunkIndex).toBe(9)
    expect(Array.from(decoded?.pcm ?? [])).toEqual([1, -2, 3])
  })
})
