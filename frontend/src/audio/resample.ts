export interface ResampleState {
  offset: number
  prev: number
  lp: number
}

export function createResampleState(): ResampleState {
  return { offset: 0, prev: 0, lp: 0 }
}

export function floatToInt16(samples: ArrayLike<number>): Int16Array {
  const pcm = new Int16Array(samples.length)
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index] ?? 0))
    pcm[index] = Math.round(clamped * 32767)
  }
  return pcm
}

export function downmix(channels: Float32Array[]): Float32Array {
  const length = channels[0]?.length ?? 0
  const mono = new Float32Array(length)
  if (!channels.length || length === 0) return mono
  const scale = 1 / channels.length
  for (let channel = 0; channel < channels.length; channel += 1) {
    const data = channels[channel]
    for (let index = 0; index < length; index += 1) mono[index] += (data[index] ?? 0) * scale
  }
  return mono
}

/** 流式线性插值，块与块之间保留上一个采样，避免在 128 帧边界处断开。 */
export function resampleBlock(
  input: Float32Array,
  inputRate: number,
  outputRate: number,
  state: ResampleState,
): Int16Array {
  if (input.length === 0 || inputRate <= 0 || outputRate <= 0) return new Int16Array(0)
  const cutoff = Math.min(inputRate, outputRate) * 0.45
  const alpha = inputRate === outputRate ? 1 : 1 - Math.exp((-2 * Math.PI * cutoff) / inputRate)
  const filtered = new Float32Array(input.length)
  let lowpass = state.lp
  for (let index = 0; index < input.length; index += 1) {
    lowpass += alpha * ((input[index] ?? 0) - lowpass)
    filtered[index] = lowpass
  }
  state.lp = lowpass
  if (inputRate === outputRate) {
    state.prev = filtered[filtered.length - 1] ?? 0
    return floatToInt16(filtered)
  }
  const step = inputRate / outputRate
  const output: number[] = []
  let offset = state.offset
  const previous = state.prev
  while (offset < filtered.length - 1 || (offset < 0 && filtered.length > 0)) {
    const index = Math.floor(offset)
    const frac = offset - index
    let left: number
    let right: number
    if (index < 0) {
      left = previous
      right = filtered[0] ?? 0
    } else if (index + 1 < filtered.length) {
      left = filtered[index] ?? 0
      right = filtered[index + 1] ?? 0
    } else {
      break
    }
    output.push(left + (right - left) * frac)
    offset += step
    if (output.length > input.length * 4 + 8) break
  }
  state.prev = filtered[filtered.length - 1] ?? previous
  state.offset = offset - filtered.length
  return floatToInt16(output)
}
