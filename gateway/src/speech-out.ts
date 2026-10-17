// Очередь фраз ответа Клона: ядро шлёт звук фраз кусками по мере синтеза, а шлюз раз в 20 мс
// берёт отсюда по кадру (pull) и сообщает ядру, что фактически звучит:
//   первый кадр ответа → playback_started; последний кадр фразы после phrase_end_marker →
//   phrase_ended; конец последней фразы после response_end → playback_finished.
// Если звук текущей фразы ещё не пришёл, pull ждёт (null), а не объявляет фразу законченной.
// Чистая логика без таймера и Discord: таймер и Player — в core-client.ts.

import { FRAME_MS, FRAME_SAMPLES } from "./mixer.ts";
import type { Message } from "./core-protocol.ts";

interface Phrase {
  frames: Int16Array[];
  // Хвост короче кадра: ждёт следующего куска или конца фразы.
  rest: Int16Array;
  ended: boolean;
}

interface Response {
  id: number;
  phrases: Map<number, Phrase>;
  // Номер звучащей (или ожидаемой) фразы.
  current: number;
  started: boolean;
  // response_end: новых фраз не будет.
  ended: boolean;
  // stop_after_phrase: после текущей фразы — конец.
  stopAfter: boolean;
}

export interface Pulled {
  frame: Int16Array | null;
  events: Message[];
}

export class SpeechOut {
  #queue: Response[] = [];
  // Ответы, остановленные ядром: запоздавший звук их фраз выбрасывается.
  #dropped = new Set<number>();

  get playing(): boolean {
    return this.#queue[0]?.started ?? false;
  }

  phraseBegin(responseId: number, phrase: number): void {
    this.#phrase(responseId, phrase);
  }

  audio(responseId: number, phrase: number, pcm: Int16Array): void {
    const p = this.#phrase(responseId, phrase);
    if (!p || p.ended) return;
    const all = new Int16Array(p.rest.length + pcm.length);
    all.set(p.rest);
    all.set(pcm, p.rest.length);
    let offset = 0;
    for (; offset + FRAME_SAMPLES <= all.length; offset += FRAME_SAMPLES) {
      p.frames.push(all.slice(offset, offset + FRAME_SAMPLES));
    }
    p.rest = all.slice(offset);
  }

  phraseEndMarker(responseId: number, phrase: number): void {
    const p = this.#phrase(responseId, phrase);
    if (!p) return;
    if (p.rest.length) {
      // Последний неполный кадр добивается тишиной.
      const last = new Int16Array(FRAME_SAMPLES);
      last.set(p.rest);
      p.frames.push(last);
      p.rest = new Int16Array(0);
    }
    p.ended = true;
  }

  responseEnd(responseId: number): void {
    const r = this.#response(responseId);
    if (r) r.ended = true;
  }

  // Договорить звучащую фразу, остальные выбросить. Если ответ ещё не звучал — выбросить целиком.
  stopAfterPhrase(responseId: number, now: number): Message[] {
    const r = this.#queue.find((x) => x.id === responseId);
    if (!r) return [];
    if (!r.started) {
      this.#remove(r);
      return [finished(r.id, now, true)];
    }
    r.stopAfter = true;
    for (const n of r.phrases.keys()) if (n > r.current) r.phrases.delete(n);
    return [];
  }

  // Замолчать сейчас: очередь очищается целиком.
  stopNow(now: number): Message[] {
    const events: Message[] = [];
    for (const r of this.#queue) {
      if (r.started) events.push(finished(r.id, now, true));
      this.#dropped.add(r.id);
    }
    this.#queue = [];
    return events;
  }

  pull(now: number): Pulled {
    const events: Message[] = [];
    for (;;) {
      const r = this.#queue[0];
      if (!r) return { frame: null, events };
      const p = r.phrases.get(r.current);
      if (!p) {
        // Фраз больше не будет — ответ закончился (или был пустым).
        if (r.ended && ![...r.phrases.keys()].some((n) => n > r.current)) {
          this.#remove(r);
          events.push(finished(r.id, now, false));
          continue;
        }
        return { frame: null, events };
      }
      const frame = p.frames.shift();
      if (!frame) {
        if (!p.ended) return { frame: null, events };
        // Пустая фраза: сразу к следующей.
        this.#endPhrase(r, now, events);
        continue;
      }
      if (!r.started) {
        r.started = true;
        events.push({ type: "playback_started", response_id: r.id, phrase: r.current, t: now });
      }
      if (p.ended && p.frames.length === 0) this.#endPhrase(r, now + FRAME_MS, events);
      return { frame, events };
    }
  }

  #endPhrase(r: Response, t: number, events: Message[]): void {
    if (r.started) events.push({ type: "phrase_ended", response_id: r.id, phrase: r.current, t });
    r.phrases.delete(r.current);
    r.current++;
    const more = [...r.phrases.keys()].some((n) => n >= r.current);
    if (r.stopAfter || (r.ended && !more)) {
      this.#remove(r);
      events.push(finished(r.id, t, r.stopAfter));
    }
  }

  #response(id: number): Response | null {
    if (this.#dropped.has(id)) return null;
    let r = this.#queue.find((x) => x.id === id);
    if (!r) {
      r = { id, phrases: new Map(), current: 0, started: false, ended: false, stopAfter: false };
      this.#queue.push(r);
    }
    return r;
  }

  #phrase(responseId: number, n: number): Phrase | null {
    const r = this.#response(responseId);
    if (!r || n < r.current || (r.stopAfter && n > r.current)) return null;
    let p = r.phrases.get(n);
    if (!p) {
      p = { frames: [], rest: new Int16Array(0), ended: false };
      r.phrases.set(n, p);
    }
    return p;
  }

  #remove(r: Response): void {
    this.#queue = this.#queue.filter((x) => x !== r);
    this.#dropped.add(r.id);
  }
}

function finished(responseId: number, t: number, interrupted: boolean): Message {
  return { type: "playback_finished", response_id: responseId, t, interrupted };
}
