# 语音对话

浏览器麦克风采集后，经服务端做流式识别，把**一句说完的话**交给兼容 OpenAI 的语言模型，边生成文字边切成短句，再交给已有的流式语音合成，由浏览器按顺序连续播放。界面同时显示流式回答。

默认是「开始说话 / 说完了」，不是全双工自动打断。

## 架构

```text
麦克风 AudioWorklet
  -> 真实重采样为 PCM16LE / 16 kHz / 单声道
  -> 二进制 WebSocket
  -> ASR 适配器（vLLM Realtime，或官方 Qwen 流式 worker）
  -> 最终一句用户话
  -> LLM SSE（只要正文，不要思维链和工具调用）
  -> 标点 / 长度 / 静默超时切句（不挡住文字显示）
  -> TTS PCM 流，边到边转发
  -> Web Audio 按采样帧排队播放
```

识别采样率固定 16000。录音处理脚本是独立文件，避免生产压缩把函数改名后采集模块调用失败、出现 0 帧。合成采样率不要假定与识别相同。`stream=true` 时上游返回 `text/event-stream`，PCM 在 `speech.audio.delta` 的 base64 `audio` 字段里，并带 `sample_rate`。`stream=false` 才是裸 `audio/pcm`。两种都按块解码后立即播放。

说话过程中，只要上游发来 `transcription.delta`，文字会马上显示，不必等「说完了」。开口之前的静音会一直等，短促噪声也不会被当成已经说过话。连续人声之后，大约 1.4 秒的停顿才结束这一句，并最多发起一次回答。没有识别到文字时不会拿空白去问模型。vLLM 在识别已经开始后会忽略再次 `commit`，因此不能靠周期提交造出增量。若部署只在最终 `commit` 之后才出字，那就不是边说边出的真流式。可用：

```bash
cd backend && python scripts/probe_realtime_partials.py
```

退出码 2 表示结束之后才有字。本环境连不上 `120.48.180.39`，所以这里没有测到该主机停录前的 partial。

## 本地启动

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example ../.env
# 只在服务器的 .env 里填写 AUDIO_LLM_API_KEY。仓库里的值是空的。
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

另一个终端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。开发服务器把 `/api` 和 WebSocket 代理到 `127.0.0.1:8080`。本机 localhost 可以使用麦克风。

## 离线演示

这不是线上模型的成功结果。

```bash
APP_MODE=mock uvicorn app.main:app --host 0.0.0.0 --port 8080
```

界面会出现明确横幅：识别、回答和声音都是本地模拟。演示不访问 `120.48.180.39`。

## 环境变量

完整列表见 `.env.example`。密钥只放在服务端：

| 变量 | 作用 |
| --- | --- |
| `AUDIO_LLM_API_KEY` | 语言模型的 Bearer。空字符串表示未配置，不会把密钥写进日志或前端 |
| `ASR_AUTHORIZATION` / `ASR_COOKIE` | 识别上游的可选鉴权 |
| `TTS_AUTHORIZATION` / `TTS_COOKIE` | 合成上游的可选鉴权 |
| `APP_AUTH_TOKEN` | 本服务自己的 WebSocket 门禁，不是上游密钥 |
| `VITE_APP_AUTH_TOKEN` | 仅当需要上一项时由前端发送。Vite 变量会进浏览器，不能拿来放上游密钥 |
| `CORS_ORIGINS` | 允许的浏览器来源，逗号分隔 |

默认上游：

- 识别文件接口：`http://120.48.180.39:10002/v1/audio/transcriptions`，模型 `Qwen3-ASR-1.7B`
- 识别实时候选：`ws://120.48.180.39:10002/v1/realtime`
- 语言模型：`http://120.48.180.39:20002/v1/chat/completions`，模型 `Qwen3.8-Flash`，请求里 `stream: true`
- 语音合成：`POST http://120.48.180.39:10001/v1/audio/speech`，模型 `/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`，`voice=vivian`，`language` 可配，`stream=true`，`response_format=pcm`

## Docker

```bash
cp .env.example .env
# 编辑 .env
docker compose up --build
```

页面在 `http://localhost:8088`。后端宿主机端口是 `BACKEND_HOST_PORT`，默认 `8080`，容器内仍监听 8080。远程部署不要用明文 HTTP 打开麦克风。

## 反向代理和 HTTPS

浏览器只在安全上下文里开放麦克风：localhost，或 HTTPS。HTTPS 页面必须使用 WSS。

`deploy/nginx.https.conf` 和 `frontend/nginx.conf` 对 `/api/ws` 做了这些设置：

- `proxy_http_version 1.1`
- `Upgrade` 与 `Connection "upgrade"`
- `proxy_buffering off`
- `proxy_cache off`
- 读写超时 3600 秒

不要给 WebSocket 或 SSE 开代理缓冲，否则识别、回答和合成都会被攒成一整包。

## 识别适配器

### realtime（默认）

按 vLLM Realtime 文档实现，而不是反复上传越来越长的音频文件：

1. 等待 `session.created`
2. 发送 `session.update`，带上模型名
3. 发送初始 `input_audio_buffer.commit`
4. 发送 `input_audio_buffer.append`，内容是 base64 的 PCM16 16 kHz 单声道
5. 说完后发送 `input_audio_buffer.commit`，`final: true`
6. 读取 `transcription.delta` / `transcription.done`

若服务端返回 `model_not_found` 或 `model_not_validated`，会话收到 `asr.unsupported`，不会假装识别成功。官方 vLLM 示例用的是 Voxtral。仅因为 vLLM 有这条路由，不能说明当前部署的 Qwen3-ASR 可用。

### qwen_streaming

独立进程调用官方接口：

- `Qwen3ASRModel.LLM(...)`
- `init_streaming_state`
- `streaming_transcribe`
- `finish_streaming_transcribe`

每个 `stream_id` 有自己的 state，按顺序处理。`state.text` 是当前全文，界面用替换而不是追加。官方流式没有词级时间戳。

依赖和 API 进程分开安装：

```bash
python3 -m venv /opt/qwen-asr-venv
/opt/qwen-asr-venv/bin/pip install -r backend/requirements-qwen-asr.txt
```

然后设置 `ASR_PROVIDER=qwen_streaming` 和 `QWEN_ASR_WORKER_PYTHON=/opt/qwen-asr-venv/bin/python`。没装 `qwen-asr` 时 worker 返回 `dependency_missing`，不会编造成功结果。

### degraded_file

只有 `ASR_PROVIDER=degraded_file` 才会走文件接口。这不是真流式。`ASR_DEGRADED_CHUNK_SEC=0` 时只在说完后上传一次。大于 0 时会按间隔重传越来越长的整段，事件里标 `cumulative_file_reupload` 和 `degraded: true`。`auto` 失败后也不会偷偷改用这个模式。

## 对话行为

- 一句最终用户话只发起一次语言模型请求。
- 文字增量立刻显示。切句按句号、问号、叹号、分号、换行、长度和静默超时进行，超时补切不会挡住已经出现的字。
- 朗读会去掉思维链、围栏代码、行内代码和 Markdown 标记。代码内容不朗读。
- 合成按句顺序开始，音频块马上转发，并用播放队列做背压，不把整段回答合成完再出声。
- 「停止回答」会取消语言模型、合成、排队音频和浏览器里已经排好的播放，并丢掉旧轮次的迟到事件。
- 被打断的助手回复会写进会话历史，并带上「回答已中断，未说完」。
- 识别失败、回答中途失败、只有合成失败、超时、坏音频和队列上限都有单独事件。
- 断开连接时会取消任务并关掉识别会话。

## 真实上游兼容性

本环境再次连接 `120.48.180.39:10001` 时连接超时，所以下面的线上结论来自已授权的部署实测，解析器用相同的 SSE 形状做了回归。没有使用用户录音。

| 项目 | 代码状态 | 线上情况 |
| --- | --- | --- |
| vLLM Realtime 与 `Qwen3-ASR-1.7B` | 握手成功后才发 `speech.started`；浏览器在此之前只缓存音频 | 部署实测该路由可用。浏览器麦克风仍需在本机确认 |
| 官方 `qwen-asr` 流式三接口 | worker 会真实调用这些方法 | 本环境未安装权重 |
| 文件转写 | 仅显式降级 | 未当作默认路径 |
| 语言模型 SSE | 只取可见正文 | 部署实测已通过 |
| 流式合成 | `speech.audio.delta` 的 base64 PCM 会按 `sample_rate` 播放；裸 PCM 仍然支持 | `stream=true` 实际是 `text/event-stream`，没有采样率响应头 |

人工要检查时，可以在确认允许访问该机器后执行：

```bash
cd backend
python -m app.probe_realtime --confirm-live
```

它只做 `session.created` 和 `session.update`，不发送音频，也不发送会开始转写的 commit。没有正向回执时，结果仍然不能当成识别可用。测试和启动都不会运行它。

## 已知限制

- 没有数据库，历史只留在当前连接的内存里。
- 没有词级时间戳。
- 不支持播报中自动抢话。要插话，先点「停止回答」。
- 识别部分结果会被替换；不要把两次 partial 拼在一起。
- 文件转写降级默认关闭。
- 采样率以帧里的元数据为准。合成默认 24000，识别是 16000。

## 测试

```bash
cd backend && python -m pytest
cd frontend && npm test && npm run typecheck && npm run build
```

后端覆盖 SSE 拆包、切句补刷和超时、取消与旧轮次隔离、识别事件、断开清理。前端覆盖重采样、采样帧对齐和二进制帧。没有在浏览器里对真实麦克风或线上模型做端到端通话。
