<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue"
import { formatLevel, micErrorMessage, microphoneBlockReason, startCapture, type CaptureStats } from "./audio/capture"
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
    receivedPeakText.value = formatLevel(Number(event.peak || 0))
    if (receivedSamples.value > 16000 && Number(event.peak || 0) < 0.008) {
      silenceHint.value = "服务端收到的声音一直很小。请检查是否选错麦克风，或系统是否把麦克风静音了。"
    }
  } else if (type === "asr.partial" || type === "asr.final") {
    const text = String(event.text || "")
    if (text) draft.value = text
    if (type === "asr.final" && event.awaiting_edit) composer.value = text
    if (type === "asr.final") micStatus.value = text ? "这句话已结束" : "没有识别到内容"
  } else if (type === "turn.started") {
    const id = `user-${event.turn_id}`
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
  receivedPeakText.value = "0.0"
  captureState.value = ""
  silenceHint.value = ""
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
        if (stats.level >= previousPeak) peakText.value = formatLevel(stats.level)
        if (sampleCount.value > 16000 && Number(peakText.value) < 0.8) {
          silenceHint.value = "采集电平一直很低。请换一个麦克风，或检查系统有没有把麦克风静音。"
        }
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
  socket?.send({ type: "speech.start", auto_submit: autoSend.value })
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
  socket?.send({ type: "speech.finish", auto_submit: autoSend.value })
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
  socket?.send({ type: "text.submit", text })
  composer.value = ""
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
        <p class="live-text" data-testid="live-transcript-text">{{ draft || "开口后，识别文字会显示在这里" }}</p>
        <p class="hint">采集 {{ frameCount }} 帧 / {{ sampleCount }} 样本 · 已发送 {{ sentFrames }} 帧 · 服务端收到 {{ receivedSamples }} 样本</p>
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
        <textarea v-model="composer" placeholder="输入文字，或在关闭自动发送后修改识别结果" @keydown="onKeydown"></textarea>
        <button type="button" :disabled="!connected || !composer.trim()" @click="submitText">发送</button>
      </div>
    </section>
  </main>
</template>
