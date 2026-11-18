<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue"
import { formatLevel, micErrorMessage, microphoneBlockReason, nextSilenceHint, startCapture, type CaptureStats } from "./audio/capture"
import { PcmPlayer } from "./audio/playback"
import { AudioSendGate } from "./audio/sendGate"
import { VoiceSocket, type ServerEvent } from "./net/session"

type Role = "user" | "assistant"
type Status = "complete" | "streaming" | "interrupted" | "error"
interface ChatMessage {
  id: string
  role: Role
  text: string
  status: Status
}

const phase = ref("正在连接")
const connected = ref(false)
const listening = ref(false)
const answering = ref(false)
const demoLabel = ref("")
const draft = ref("")
const composer = ref("")
const autoSend = ref(true)
const reviewGrowth = ref(false)
const correctOn = ref(true)
const rawFinal = ref("")
const correctedText = ref("")
const suggestText = ref("")
const correctNote = ref("")
let composerDirty = false
let correctUtterance = ""
const forwardedSamples = ref(0)
const heldSamples = ref(0)
const muted = ref(false)
const volume = ref(80)
const errorText = ref("")
const note = ref("说话时会显示识别。停顿或点「说完了」后才回答，播报时不会抢话。")
const asrNote = ref("")
const messages = ref<ChatMessage[]>([])
const captureState = ref("")
const frameCount = ref(0)
const sentFrames = ref(0)
const sampleCount = ref(0)
const levelText = ref("0.0")
const peakText = ref("0.0")
const receivedSamples = ref(0)
const receivedPeakText = ref("0.0")
const micStatus = ref("")
const silenceHint = ref("")
const devices = ref<Array<{ id: string; label: string }>>([])
const deviceId = ref("")
const currentDevice = ref("尚未选择")
let socket: VoiceSocket | null = null
let stopMic: (() => Promise<void>) | null = null
let sendGate: AudioSendGate | null = null
let seqGuard = 0
let sealedUtterance = ""
let asrReady = false
let asrFailedMessage = ""
let readyWaiters: Array<() => void> = []

const player = new PcmPlayer(
  1.8,
  (hold) => socket?.send({ type: "playback.hold", hold }),
  (bytes) => socket?.send({ type: "playback.ack", bytes }),
)
const canStart = computed(() => connected.value && !listening.value && !answering.value)
const canFinish = computed(() => connected.value && listening.value)
const canStop = computed(() => connected.value && answering.value)
const secureHint = computed(() =>
  microphoneBlockReason(window.isSecureContext, !!navigator.mediaDevices?.getUserMedia),
)

function pushError(message: string) {
  if (message) errorText.value = message
}

function upsertAssistant(turnId: string, text: string, status: Status) {
  const found = messages.value.find((item) => item.id === turnId)
  if (found) {
    found.text = text
    found.status = status
    return
  }
  messages.value.push({ id: turnId, role: "assistant", text, status })
}

function onEvent(event: ServerEvent) {
  const type = String(event.type || "")
  if (type === "session.ready") {
    demoLabel.value = String(event.demo_label || "")
    const asr = (event.asr || {}) as { note?: string; sample_rate?: number; provider?: string }
    const tts = (event.tts || {}) as { sample_rate?: number }
    asrNote.value = `识别 ${asr.provider || ""} · ${asr.sample_rate || 16000} Hz，合成 ${tts.sample_rate || 24000} Hz。两者采样率各自配置。`
    if (asr.note) note.value = String(asr.note)
  } else if (type === "state") {
    phase.value = String(event.label || event.phase || "")
    const name = String(event.phase || "")
    listening.value = name === "listening"
    answering.value = name === "answering"
  } else if (type === "speech.started" || type === "asr.ready") {
    markAsrReady()
    micStatus.value = "正在等待开口"
  } else if (type === "listen.activity") {
    const stage = String(event.stage || "")
    if (stage === "waiting") micStatus.value = "正在等待开口"
    if (stage === "speech") micStatus.value = "听到说话了，识别会跟着更新"
    if (stage === "pause") micStatus.value = "停顿中，可以继续说"
  } else if (type === "audio.progress") {
    receivedSamples.value = Number(event.samples || 0)
    forwardedSamples.value = Number(event.forwarded_samples || 0)
    heldSamples.value = Number(event.held_samples || 0)
    receivedPeakText.value = formatLevel(Number(event.peak || 0))
    silenceHint.value = nextSilenceHint(silenceHint.value, "server", receivedSamples.value, Number(event.peak || 0))
  } else if (type === "asr.partial" || type === "asr.final") {
    const utterance = String(event.utterance_id || "")
    if (utterance && utterance === sealedUtterance && type === "asr.partial") return
    if (type === "asr.final" && utterance && utterance === sealedUtterance) return
    const text = String(event.text || "")
    if (text) draft.value = text
    if (type === "asr.final") {
      if (utterance) sealedUtterance = utterance
      correctUtterance = utterance
      rawFinal.value = text
      correctedText.value = ""
      suggestText.value = ""
      correctNote.value = ""
      composerDirty = false
      if (event.awaiting_edit) composer.value = text
      micStatus.value = event.review_reason
        ? `${event.review_reason}。全文在输入框里，改完再发送，这里不会替你删字。`
        : text
          ? "这句话已结束"
          : "没有识别到内容"
    }
  } else if (type === "asr.correcting") {
    const utterance = String(event.utterance_id || "")
    if (utterance !== correctUtterance) return
    micStatus.value = "正在校正。这句话已经结束，不会挡住下一次开口，也不能保证马上完成。"
  } else if (type === "asr.corrected") {
    const utterance = String(event.utterance_id || "")
    if (utterance !== correctUtterance) return
    const original = String(event.original || rawFinal.value || "")
    const text = String(event.text || original)
    const reason = String(event.reason || "")
    const elapsed = Number(event.elapsed_ms || 0)
    rawFinal.value = original
    if (text && text !== original) correctedText.value = text
    else correctedText.value = ""
    if (event.suggest_only && text && text !== original && !composerDirty) suggestText.value = text
    const changeText = Array.isArray(event.changes)
      ? event.changes
          .map((item) => {
            const change = item as { kind?: string; from?: string; to?: string }
            if (change.kind === "glossary") return `${change.from} → ${change.to}`
            if (change.kind === "punctuation") return "调整了标点"
            return ""
          })
          .filter(Boolean)
          .join("；")
      : ""
    correctNote.value = [correctReason(reason), elapsed ? `用时 ${elapsed} 毫秒。` : "", changeText ? `${changeText}。` : ""].filter(Boolean).join("")
    micStatus.value = correctReason(reason)
  } else if (type === "turn.started") {
    const id = `user-${event.turn_id}`
    if (messages.value.some((item) => item.id === id)) return
    messages.value.push({ id, role: "user", text: String(event.text || ""), status: "complete" })
    micStatus.value = "这句话已放入对话"
    answering.value = true
  } else if (type === "llm.delta" || type === "llm.done") {
    upsertAssistant(String(event.turn_id), String(event.text || ""), type === "llm.done" ? "complete" : "streaming")
  } else if (type === "llm.error") {
    upsertAssistant(String(event.turn_id), String(event.partial_text || ""), "error")
    pushError(String(event.message || "回答中断"))
  } else if (type === "turn.cancelled") {
    const id = String(event.turn_id || "")
    const found = messages.value.find((item) => item.id === id)
    if (found) found.status = "interrupted"
    player.stop(id)
  } else if (type === "playback.stop") {
    player.stop(String(event.turn_id || ""))
  } else if (type === "conversation.cleared") {
    messages.value = []
    draft.value = ""
    composer.value = ""
  } else if (type === "asr.unsupported" || type === "asr.failed" || type === "tts.failed" || type === "timeout" || type === "queue.limit" || type === "audio.malformed" || type === "error") {
    pushError(String(event.message || "出现问题"))
    if (type === "asr.failed" || type === "asr.unsupported") {
      listening.value = false
      if (!asrReady) markAsrFailed(String(event.message || "识别没有就绪"))
      void stopMicNow()
    }
  }
}

function correctReason(reason: string): string {
  if (reason === "ok") return "已用修正稿。只接受了标点或词表，没有补写没识别到的内容。"
  if (reason === "mock") return "离线校正只补了句末标点，不是线上模型。"
  if (reason === "glossary") return "已按词表替换，其余保持原稿。"
  if (reason === "original") return "校正后与原稿一致。"
  if (reason === "fallback_timeout") return "校正超时，使用原稿。"
  if (reason === "fallback_empty") return "校正没有返回文字，使用原稿。"
  if (reason === "fallback_invalid") return "校正结果不能用，使用原稿。"
  if (reason === "fallback_unconfigured") return "没有配置校正用的语言模型，使用原稿。"
  return "使用原稿。"
}

function shownTranscript(): string {
  return correctedText.value || draft.value
}

function adoptSuggestion() {
  if (!suggestText.value) return
  composer.value = suggestText.value
  composerDirty = true
}

function resetCaptureWait() {
  asrReady = false
  asrFailedMessage = ""
  readyWaiters = []
  frameCount.value = 0
  sentFrames.value = 0
  sampleCount.value = 0
  levelText.value = "0.0"
  peakText.value = "0.0"
  receivedSamples.value = 0
  forwardedSamples.value = 0
  heldSamples.value = 0
  receivedPeakText.value = "0.0"
  captureState.value = ""
  silenceHint.value = ""
  sealedUtterance = ""
  correctUtterance = ""
  rawFinal.value = ""
  correctedText.value = ""
  suggestText.value = ""
  correctNote.value = ""
  composerDirty = false
  draft.value = ""
  micStatus.value = "正在打开麦克风"
}

function markAsrReady() {
  if (asrReady || asrFailedMessage) return
  asrReady = true
  sendGate?.release()
  readyWaiters.splice(0).forEach((wake) => wake())
}

function markAsrFailed(message: string) {
  if (asrReady || asrFailedMessage) return
  asrFailedMessage = message
  sendGate?.close()
  readyWaiters.splice(0).forEach((wake) => wake())
}

function waitForAsr(timeoutMs: number): Promise<boolean> {
  if (asrReady) return Promise.resolve(true)
  if (asrFailedMessage) return Promise.resolve(false)
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => resolve(asrReady), timeoutMs)
    readyWaiters.push(() => {
      window.clearTimeout(timer)
      resolve(asrReady && !asrFailedMessage)
    })
  })
}

async function stopMicNow() {
  const stop = stopMic
  stopMic = null
  if (stop) await stop()
}

function onAudio(frame: { turnId: string; pcm: Int16Array; sampleRate: number; channels: number }) {
  player.enqueue(frame.pcm, frame.sampleRate, frame.channels, frame.turnId)
}

async function connect() {
  socket = new VoiceSocket(onEvent, onAudio, (status) => {
    connected.value = status === "open"
    if (status === "connecting") phase.value = "正在连接"
    if (status === "closed") {
      phase.value = "连接已断开"
      listening.value = false
      answering.value = false
    }
  })
  socket.connect()
}

async function beginSpeech() {
  errorText.value = ""
  if (secureHint.value) {
    pushError(secureHint.value)
    return
  }
  player.stop()
  const generation = ++seqGuard
  resetCaptureWait()
  sendGate = new AudioSendGate((frame) => {
    if (generation !== seqGuard) return
    sentFrames.value += 1
    socket?.sendAudio(frame)
  })
  const playbackReady = player.resume()
  try {
    stopMic = await startCapture({
      deviceId: deviceId.value,
      onFrame: (frame) => sendGate?.push(frame),
      onError: (message) => {
        if (generation === seqGuard) pushError(message)
      },
      onStats: (stats: CaptureStats) => {
        if (generation !== seqGuard) return
        captureState.value = stats.contextState
        frameCount.value = stats.frames
        sampleCount.value = stats.samples
        levelText.value = formatLevel(stats.level)
        const previousPeak = Number(peakText.value) / 100
        const peak = Math.max(stats.level, previousPeak)
        if (stats.level >= previousPeak) peakText.value = formatLevel(stats.level)
        silenceHint.value = nextSilenceHint(silenceHint.value, "mic", sampleCount.value, peak)
      },
    })
    await playbackReady
    await refreshDevices()
  } catch (error) {
    seqGuard += 1
    sendGate.close()
    await stopMicNow()
    pushError(micErrorMessage(error))
    return
  }
  if (generation !== seqGuard) return
  socket?.send({
    type: "speech.start",
    auto_submit: autoSend.value,
    review_growth: reviewGrowth.value,
    correct: correctOn.value,
  })
  listening.value = true
  phase.value = "正在聆听"
  micStatus.value = "正在等待开口"
  window.setTimeout(() => {
    if (generation !== seqGuard || !listening.value) return
    if (sampleCount.value === 0) {
      pushError("录音模块没有交出采样。请看浏览器是否允许麦克风；若刚允许，请再点一次「开始说话」。")
    }
  }, 1500)
  window.setTimeout(() => {
    if (generation !== seqGuard || !listening.value) return
    if (micStatus.value === "正在等待开口") micStatus.value = "仍在等待开口，可以直接说"
  }, 20000)
}

async function finishSpeech() {
  const generation = seqGuard
  await stopMicNow()
  if (generation !== seqGuard) return
  const ready = await waitForAsr(8000)
  if (!ready || asrFailedMessage) {
    pushError(asrFailedMessage || "识别通道没有就绪，这次录音已取消。")
    sendGate?.close()
    socket?.send({ type: "speech.cancel" })
    listening.value = false
    phase.value = "待命"
    return
  }
  const leftover = sendGate?.close() ?? []
  for (const frame of leftover) socket?.sendAudio(frame)
  socket?.send({ type: "speech.finish", auto_submit: autoSend.value, review_growth: reviewGrowth.value })
  listening.value = false
  phase.value = "正在识别"
  micStatus.value = `已采集 ${frameCount.value} 帧 / ${sampleCount.value} 样本`
}

async function cancelSpeech() {
  seqGuard += 1
  sendGate?.close()
  await stopMicNow()
  socket?.send({ type: "speech.cancel" })
  listening.value = false
  draft.value = ""
  micStatus.value = "已取消"
  phase.value = "待命"
}

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return
  const listed = await navigator.mediaDevices.enumerateDevices()
  const inputs = listed.filter((item) => item.kind === "audioinput")
  devices.value = inputs.map((item, index) => ({
    id: item.deviceId,
    label: item.label || `麦克风 ${index + 1}`,
  }))
  const selected = devices.value.find((item) => item.id === deviceId.value) || devices.value[0]
  if (selected) {
    deviceId.value = selected.id
    currentDevice.value = selected.label
  }
}

function submitText() {
  const text = composer.value.trim()
  if (!text) return
  correctUtterance = ""
  socket?.send({ type: "text.submit", text })
  composer.value = ""
  composerDirty = false
  suggestText.value = ""
}

function stopAnswer() {
  socket?.send({ type: "answer.stop" })
  player.stop()
  answering.value = false
}

function clearConversation() {
  socket?.send({ type: "conversation.clear" })
  player.stop()
}

function toggleMute() {
  muted.value = !muted.value
  player.setMuted(muted.value)
}

function onVolume() {
  player.setVolume(volume.value / 100)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    submitText()
  }
}

onMounted(() => {
  player.setVolume(volume.value / 100)
  void refreshDevices()
  void connect()
})

onBeforeUnmount(() => {
  void stopMic?.()
  socket?.close()
})
</script>

<template>
  <main class="app">
    <section class="panel">
      <div class="brand">
        <div>
          <h1>语音对话</h1>
          <p>说完再识别，回答一边显示一边朗读。</p>
        </div>
        <div class="lamp" :class="{ on: answering, talk: listening }"></div>
      </div>
      <div>
        <strong>{{ phase }}</strong>
        <p class="hint">{{ note }}</p>
      </div>
      <div class="actions">
        <button type="button" :disabled="!canStart" @click="beginSpeech">开始说话</button>
        <button type="button" class="secondary" :disabled="!canFinish" @click="finishSpeech">说完了</button>
        <button type="button" class="warn" :disabled="!canStop" @click="stopAnswer">停止回答</button>
      </div>
      <div class="actions">
        <button type="button" class="secondary" :disabled="!listening" @click="cancelSpeech">取消这次</button>
        <button type="button" class="secondary" :disabled="!connected" @click="toggleMute">{{ muted ? "取消静音" : "静音" }}</button>
        <button type="button" class="secondary" :disabled="!connected" @click="clearConversation">清空对话</button>
      </div>
      <label class="volume">音量
        <input v-model.number="volume" type="range" min="0" max="100" @input="onVolume" />
      </label>
      <label class="volume">麦克风
        <select v-model="deviceId" :disabled="listening">
          <option v-if="devices.length === 0" value="">默认麦克风</option>
          <option v-for="item in devices" :key="item.id" :value="item.id">{{ item.label }}</option>
        </select>
      </label>
      <p class="hint">当前麦克风：{{ currentDevice }}</p>
      <label class="check">
        <input v-model="autoSend" type="checkbox" />
        识别完成后自动发送
      </label>
      <label class="check">
        <input v-model="reviewGrowth" type="checkbox" />
        停口后识别仍变长时，先让我改再发送
      </label>
      <label class="check">
        <input v-model="correctOn" type="checkbox" :disabled="listening" />
        终稿后校正标点和词表
      </label>
      <p class="hint">校正默认打开。它在这句话结束之后单独请求一次，不跟着每个字走，也不保证马上完成。失败就用原稿。它不能补回没识别到的声音，也不能改数字、否定和词表以外的专名。</p>
      <p class="hint">自动发送保持打开。后一个选项默认关闭；打开后，只有识别在你停止出声之后又多出文字时才放进输入框。多出来的字会原样留下，供你自己删。</p>
      <p class="hint">{{ asrNote }}</p>
      <button type="button" class="secondary" @click="connect">重新连接</button>
    </section>
    <section class="paper">
      <div v-if="secureHint" class="error">{{ secureHint }}</div>
      <div v-if="demoLabel" class="banner">{{ demoLabel }}</div>
      <div v-if="errorText" class="error">{{ errorText }}</div>
      <section class="live-transcript" data-testid="live-transcript">
        <h2>实时转写</h2>
        <p data-testid="live-transcript-status">{{ micStatus || "还没开始" }}</p>
        <p class="live-text" data-testid="live-transcript-text">{{ shownTranscript() || "开口后，识别文字会显示在这里" }}</p>
        <p v-if="rawFinal && shownTranscript() !== rawFinal" class="hint" data-testid="live-transcript-original">原稿：{{ rawFinal }}</p>
        <p v-if="correctNote" class="hint" data-testid="correct-note">{{ correctNote }}</p>
        <button v-if="suggestText" type="button" class="secondary" @click="adoptSuggestion">采用修正稿</button>
        <p class="hint">采集 {{ frameCount }} 帧 / {{ sampleCount }} 样本 · 已发送 {{ sentFrames }} 帧 · 服务端收到 {{ receivedSamples }} 样本</p>
        <p class="hint">已送识别 {{ (forwardedSamples / 16000).toFixed(1) }} 秒 · 开口后的长静音未送 {{ (heldSamples / 16000).toFixed(1) }} 秒</p>
        <p v-if="listening && sentFrames > 0 && !draft" class="hint">声音已经送出。识别服务通常要几秒才返回第一段整句，页面收到后整段替换，不会把已有的句子做成逐字动画。</p>
        <p class="hint">当前电平 {{ levelText }}% · 峰值 {{ peakText }}% · 服务端峰值 {{ receivedPeakText }}%</p>
        <div class="meter" aria-hidden="true"><span :style="{ width: Math.min(100, Number(peakText)) + '%' }"></span></div>
        <p v-if="silenceHint" class="error">{{ silenceHint }}</p>
      </section>
      <div class="messages">
        <article v-for="item in messages" :key="item.id" class="bubble" :class="item.role">
          {{ item.text || "…" }}
          <span v-if="item.status === 'interrupted'" class="badge">已中断</span>
          <span v-else-if="item.status === 'error'" class="badge">回答出错</span>
          <span v-else-if="item.status === 'streaming'" class="badge">正在生成</span>
        </article>
        <p v-if="messages.length === 0" class="meta">回答会出现在这里。正在说的话看上面的实时转写。</p>
      </div>
      <div class="composer">
        <textarea v-model="composer" placeholder="输入文字，或在关闭自动发送后修改识别结果" @input="composerDirty = true" @keydown="onKeydown"></textarea>
        <button type="button" :disabled="!connected || !composer.trim()" @click="submitText">发送</button>
      </div>
    </section>
  </main>
</template>
