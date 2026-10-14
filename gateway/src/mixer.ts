// Смешивание PCM-потоков участников в кадры по 20 мс: моно, s16, 48 кГц.
// У каждого участника своя очередь кадров с временем получения. pull(now) берёт у каждого
// по одному кадру, «созревшему» на delayMs, и складывает их с насыщением до int16.
// Если созревших кадров нет — null: плеер простаивает, а не играет тишину.

export const SAMPLE_RATE = 48_000;
export const FRAME_MS = 20;
export const FRAME_SAMPLES = (SAMPLE_RATE / 1000) * FRAME_MS; // 960

interface QueuedFrame {
  pcm: Int16Array;
  t: number;
}

export class Mixer {
  readonly #delayMs: number;
  #queues = new Map<string, QueuedFrame[]>();

  constructor(delayMs: number) {
    this.#delayMs = delayMs;
  }

  push(userId: string, pcm: Int16Array, t: number): void {
    let queue = this.#queues.get(userId);
    if (!queue) {
      queue = [];
      this.#queues.set(userId, queue);
    }
    queue.push({ pcm, t });
  }

  pull(now: number): Int16Array | null {
    let sum: Int32Array | null = null;
    for (const [userId, queue] of this.#queues) {
      const head = queue[0];
      if (!head || head.t + this.#delayMs > now) continue;
      queue.shift();
      if (queue.length === 0) this.#queues.delete(userId);
      sum ??= new Int32Array(FRAME_SAMPLES);
      const n = Math.min(FRAME_SAMPLES, head.pcm.length);
      for (let i = 0; i < n; i++) sum[i]! += head.pcm[i]!;
    }
    if (sum === null) return null;
    const out = new Int16Array(FRAME_SAMPLES);
    for (let i = 0; i < FRAME_SAMPLES; i++) {
      out[i] = Math.max(-32768, Math.min(32767, sum[i]!));
    }
    return out;
  }

  clear(): void {
    this.#queues.clear();
  }
}

// StreamType.Raw в @discordjs/voice ожидает s16le 48 кГц стерео: дублируем моно в оба канала.
export function monoToStereo(mono: Int16Array): Buffer {
  const out = Buffer.alloc(mono.length * 4);
  for (let i = 0; i < mono.length; i++) {
    out.writeInt16LE(mono[i]!, i * 4);
    out.writeInt16LE(mono[i]!, i * 4 + 2);
  }
  return out;
}

// Декодер Opus отдаёт s16le стерео; для приёма и записи сводим в моно.
export function stereoToMono(stereo: Buffer): Int16Array {
  const samples = stereo.length / 4;
  const out = new Int16Array(samples);
  for (let i = 0; i < samples; i++) {
    out[i] = (stereo.readInt16LE(i * 4) + stereo.readInt16LE(i * 4 + 2)) >> 1;
  }
  return out;
}
