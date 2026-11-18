<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue"
import { micErrorMessage, startCapture } from "./audio/capture"
import { PcmPlayer } from "./audio/playback"
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
const note = ref("按住式对话：先开始说话，说完再结束。播报时不会自动抢话。")
const asrNote = ref("")
const messages = ref<ChatMessage[]>([])
let socket: VoiceSocket | null = null
let stopMic: (() => Promise<void>) | null = null
let seqGuard = 0

const player = new PcmPlayer(
  1.8,
  (hold) => socket?.send({ type: "playback.hold", hold }),
  (bytes) => socket?.send({ type: "playback.ack", bytes }),
)
const canStart = computed(() => connected.value && !listening.value && !answering.value)
const canFinish = computed(() => connected.value && listening.value)
const canStop = computed(() => connected.value && answering.value)

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
  } else if (type === "asr.partial" || type === "asr.final") {
    draft.value = String(event.text || "")
    if (type === "asr.final" && event.awaiting_edit) composer.value = draft.value
    if (type === "asr.final" && !event.awaiting_edit) draft.value = ""
  } else if (type === "turn.started") {
    const id = `user-${event.turn_id}`
    messages.value.push({ id, role: "user", text: String(event.text || ""), status: "complete" })
    draft.value = ""
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
    if (type === "asr.failed" || type === "asr.unsupported") listening.value = false
  }
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
  try {
    await player.resume()
    const generation = ++seqGuard
    stopMic = await startCapture((frame) => {
      if (generation === seqGuard) socket?.sendAudio(frame)
    })
    socket?.send({ type: "speech.start" })
    listening.value = true
    phase.value = "正在聆听"
  } catch (error) {
    pushError(micErrorMessage(error))
  }
}

async function finishSpeech() {
  if (stopMic) await stopMic()
  stopMic = null
  socket?.send({ type: "speech.finish", auto_submit: autoSend.value })
  listening.value = false
  phase.value = "正在识别"
}

async function cancelSpeech() {
  if (stopMic) await stopMic()
  stopMic = null
  socket?.send({ type: "speech.cancel" })
  listening.value = false
  draft.value = ""
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
      <label class="check">
        <input v-model="autoSend" type="checkbox" />
        识别完成后自动发送
      </label>
      <p v-if="draft" class="draft">实时转写：{{ draft }}</p>
      <p class="hint">{{ asrNote }}</p>
      <button type="button" class="secondary" @click="connect">重新连接</button>
    </section>
    <section class="paper">
      <div v-if="demoLabel" class="banner">{{ demoLabel }}</div>
      <div v-if="errorText" class="error">{{ errorText }}</div>
      <div class="messages">
        <article v-for="item in messages" :key="item.id" class="bubble" :class="item.role">
          {{ item.text || "…" }}
          <span v-if="item.status === 'interrupted'" class="badge">已中断</span>
          <span v-else-if="item.status === 'error'" class="badge">回答出错</span>
          <span v-else-if="item.status === 'streaming'" class="badge">正在生成</span>
        </article>
        <p v-if="messages.length === 0" class="meta">还没有对话。可以打字发送，也可以开始说话。</p>
      </div>
      <div class="composer">
        <textarea v-model="composer" placeholder="输入文字，或在关闭自动发送后修改识别结果" @keydown="onKeydown"></textarea>
        <button type="button" :disabled="!connected || !composer.trim()" @click="submitText">发送</button>
      </div>
    </section>
  </main>
</template>
