function floatToInt16(samples) {
  const pcm = new Int16Array(samples.length)
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index] || 0))
    pcm[index] = Math.round(clamped * 32767)
  }
  return pcm
}

function downmix(channels) {
  const length = channels[0] ? channels[0].length : 0
  const mono = new Float32Array(length)
  if (!channels.length || length === 0) return mono
  const scale = 1 / channels.length
  for (let channel = 0; channel < channels.length; channel += 1) {
    const data = channels[channel]
    for (let index = 0; index < length; index += 1) mono[index] += (data[index] || 0) * scale
  }
  return mono
}

function resampleBlock(input, inputRate, outputRate, state) {
  if (!input.length || inputRate <= 0 || outputRate <= 0) return new Int16Array(0)
  const cutoff = Math.min(inputRate, outputRate) * 0.45
  const alpha = inputRate === outputRate ? 1 : 1 - Math.exp((-2 * Math.PI * cutoff) / inputRate)
  const filtered = new Float32Array(input.length)
  let lowpass = state.lp
  for (let index = 0; index < input.length; index += 1) {
    lowpass += alpha * ((input[index] || 0) - lowpass)
    filtered[index] = lowpass
  }
  state.lp = lowpass
  if (inputRate === outputRate) {
    state.prev = filtered[filtered.length - 1] || 0
    return floatToInt16(filtered)
  }
  const step = inputRate / outputRate
  const output = []
  let offset = state.offset
  const previous = state.prev
  while (offset < filtered.length - 1 || (offset < 0 && filtered.length > 0)) {
    const index = Math.floor(offset)
    const frac = offset - index
    let left
    let right
    if (index < 0) {
      left = previous
      right = filtered[0] || 0
    } else if (index + 1 < filtered.length) {
      left = filtered[index] || 0
      right = filtered[index + 1] || 0
    } else {
      break
    }
    output.push(left + (right - left) * frac)
    offset += step
    if (output.length > input.length * 4 + 8) break
  }
  state.prev = filtered[filtered.length - 1] || previous
  state.offset = offset - filtered.length
  return floatToInt16(output)
}

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.state = { offset: 0, prev: 0, lp: 0 }
    this.targetRate = 16000
    this.port.postMessage({ type: "ready", sampleRate })
  }

  process(inputs, outputs) {
    try {
      const outputChannels = outputs && outputs[0]
      if (outputChannels) {
        for (let channel = 0; channel < outputChannels.length; channel += 1) {
          outputChannels[channel].fill(0)
        }
      }
      const channels = inputs[0]
      if (!channels || channels.length === 0 || !channels[0] || channels[0].length === 0) {
        return true
      }
      const mono = downmix(channels)
      const pcm = resampleBlock(mono, sampleRate, this.targetRate, this.state)
      if (pcm.length > 0) {
        this.port.postMessage(pcm.buffer, [pcm.buffer])
      }
    } catch (error) {
      this.port.postMessage({ type: "error", message: String(error && error.message ? error.message : error) })
      return false
    }
    return true
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor)
