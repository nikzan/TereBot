// Тестовый шлюз без Discord: проверка ядра. Ведёт Сессию, подаёт WAV как речь участника
// в реальном темпе и «проигрывает» ответ через настоящую очередь фраз (speech-out.ts),
// отправляя ядру те же отчёты, что шлюз. Печатает ход событий и сохраняет звук ответов.
//
//   node scripts/fake-gateway.ts --wav question.wav [--barge-in-wav stop.wav --barge-in-after 2]
//
// WAV — PCM 16 бит моно 48 кГц (как пишет Recorder).

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { loadSettings, REPO_ROOT } from "../src/config.ts";
import { decodeAudio, decodeMessage, encodeAudio, encodeMessage, type Message } from "../src/core-protocol.ts";
import { FRAME_MS, FRAME_SAMPLES } from "../src/mixer.ts";
import { WavWriter, sessionDirName } from "../src/recorder.ts";
import { SpeechOut } from "../src/speech-out.ts";

const { values: args } = parseArgs({
  options: {
    wav: { type: "string" },
    "barge-in-wav": { type: "string" },
    "barge-in-after": { type: "string", default: "2" },
    name: { type: "string", default: "Петя" },
    timeout: { type: "string", default: "90" },
  },
});

const settings = loadSettings();
const userId = settings.discord.whitelist[0]!;
const start = Date.now();
const stamp = () =>
  `${new Date().toTimeString().slice(0, 8)}.${String(Date.now() % 1000).padStart(3, "0")} ` +
  `(+${((Date.now() - start) / 1000).toFixed(2)} с)`;
const log = (msg: string) => console.log(`${stamp()}  ${msg}`);

function readWav(path: string): Int16Array {
  const buf = readFileSync(path);
  if (buf.readUInt32LE(24) !== 48_000 || buf.readUInt16LE(22) !== 1) throw new Error(`${path}: нужен моно 48 кГц`);
  const data = buf.subarray(44);
  const pcm = new Int16Array(data.length / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = data.readInt16LE(i * 2);
  return pcm;
}

const ws = new WebSocket(`ws://127.0.0.1:${settings.core.port}`);
ws.binaryType = "arraybuffer";
const send = (m: Message) => ws.send(encodeMessage(m));
const speech = new SpeechOut();
const outDir = join(REPO_ROOT, "data", "probe", sessionDirName(new Date()));
mkdirSync(outDir, { recursive: true });
const wavs = new Map<number, WavWriter>();
const firstAudio = new Set<string>();
// Звук, пришедший от ядра после stop_now для оборванных этим stop_now ответов.
const stopped = new Map<number, number>();
const finished = new Set<number>();
const withAudio = new Set<number>();
const waiters: Array<{ match: (m: Message) => boolean; resolve: (m: Message) => void }> = [];

function observe(m: Message): void {
  for (const w of [...waiters]) {
    if (w.match(m)) {
      waiters.splice(waiters.indexOf(w), 1);
      w.resolve(m);
    }
  }
}

function waitFor(match: (m: Message) => boolean, timeoutS: number): Promise<Message | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), timeoutS * 1000);
    waiters.push({ match, resolve: (m) => (clearTimeout(timer), resolve(m)) });
  });
}

ws.addEventListener("message", (e) => {
  if (e.data instanceof ArrayBuffer) {
    const f = decodeAudio(Buffer.from(e.data));
    if (f.kind !== "response") return;
    withAudio.add(f.responseId);
    if (stopped.has(f.responseId)) stopped.set(f.responseId, stopped.get(f.responseId)! + 1);
    const key = `${f.responseId}/${f.phrase}`;
    if (!firstAudio.has(key)) {
      firstAudio.add(key);
      log(`ядро → звук ответа ${f.responseId}, фраза ${f.phrase}: первый кусок`);
    }
    let w = wavs.get(f.responseId);
    if (!w) wavs.set(f.responseId, (w = new WavWriter(join(outDir, `response-${f.responseId}.wav`))));
    w.write(f.pcm);
    speech.audio(f.responseId, f.phrase, f.pcm);
    return;
  }
  const m = decodeMessage(String(e.data));
  log(`ядро → ${JSON.stringify(m)}`);
  const now = Date.now();
  switch (m.type) {
    case "phrase_begin":
      speech.phraseBegin(m.response_id as number, m.phrase as number);
      break;
    case "phrase_end_marker":
      speech.phraseEndMarker(m.response_id as number, m.phrase as number);
      break;
    case "response_end":
      speech.responseEnd(m.response_id as number);
      break;
    case "stop_after_phrase":
      emit(speech.stopAfterPhrase(m.response_id as number, now));
      break;
    case "stop_now":
      for (const id of withAudio) if (!finished.has(id)) stopped.set(id, stopped.get(id) ?? 0);
      emit(speech.stopNow(now));
      break;
  }
});

function emit(events: Message[]): void {
  for (const ev of events) {
    if (ev.type === "playback_finished") finished.add(ev.response_id as number);
    log(`шлюз → ${JSON.stringify(ev)}`);
    send(ev);
    observe(ev);
  }
}

let next = 0;
setInterval(() => {
  const now = Date.now();
  while (next <= now) {
    emit(speech.pull(next).events);
    next += FRAME_MS;
  }
}, 10);

async function speak(path: string): Promise<void> {
  const pcm = readWav(path);
  const t0 = Date.now();
  send({ type: "segment_start", user_id: userId, t: t0 });
  log(`участник говорит ${path} (${(pcm.length / 48_000).toFixed(1)} с)`);
  for (let i = 0; i + FRAME_SAMPLES <= pcm.length; i += FRAME_SAMPLES) {
    const due = t0 + ((i + FRAME_SAMPLES) / FRAME_SAMPLES) * FRAME_MS;
    await new Promise((r) => setTimeout(r, Math.max(0, due - Date.now())));
    ws.send(encodeAudio({ kind: "participant", userId, t: Date.now(), pcm: pcm.subarray(i, i + FRAME_SAMPLES) }));
  }
  log("участник замолчал");
  // Как Discord: тишину клиент не шлёт, подписка закрывается через 1 с.
  setTimeout(() => send({ type: "segment_end", user_id: userId, t: Date.now() }), 1000);
}

await new Promise((r) => ws.addEventListener("open", r, { once: true }));
next = Date.now();
const session = { name: `probe-${sessionDirName(new Date())}`, t: Date.now() };
send({ type: "hello", session: null, participants: [], t: Date.now() });
send({ type: "session_start", session, t: session.t });
send({ type: "participant_joined", user_id: userId, name: args.name!, t: Date.now() });
log(`Сессия ${session.name}`);

const greeting = await waitFor((m) => m.type === "playback_finished", 30);
log(greeting ? "приветствие доиграло" : "приветствия нет за 30 с");

if (args.wav) {
  await new Promise((r) => setTimeout(r, 1000));
  await speak(args.wav);
  const started = await waitFor((m) => m.type === "playback_started", Number(args.timeout));
  if (started && args["barge-in-wav"]) {
    await new Promise((r) => setTimeout(r, Number(args["barge-in-after"]) * 1000));
    await speak(args["barge-in-wav"]);
  }
  if (started && !finished.has(started.response_id as number)) {
    await waitFor((m) => m.type === "playback_finished" && m.response_id === started.response_id, Number(args.timeout));
  }
  await new Promise((r) => setTimeout(r, 3000));
}

for (const [id, n] of stopped) log(`ответ ${id}: кусков звука после stop_now — ${n}`);
send({ type: "session_end", t: Date.now() });
for (const w of wavs.values()) w.close();
log(`звук ответов — ${outDir}`);
await new Promise((r) => setTimeout(r, 300));
process.exit(0);
