import { downmix, floatToInt16, resampleBlock } from "./resample"

export function buildWorkletSource(): string {
  return `
${downmix.toString()}
${floatToInt16.toString()}
${resampleBlock.toString()}
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.state = { offset: 0, prev: 0, lp: 0 };
    this.targetRate = 16000;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0 || !channels[0] || channels[0].length === 0) {
      return true;
    }
    const mono = downmix(channels);
    const pcm = resampleBlock(mono, sampleRate, this.targetRate, this.state);
    if (pcm.length > 0) {
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCaptureProcessor);
`
}

export function workletModuleUrl(): string {
  const blob = new Blob([buildWorkletSource()], { type: "application/javascript" })
  return URL.createObjectURL(blob)
}
