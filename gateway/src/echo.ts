// Режим эха для Эхо-теста: каждый говорящий звучит в канале с задержкой echo_delay_ms,
// одновременные голоса смешиваются. Подписчик приёмника, сам приёмник не меняет.

import { FRAME_MS, Mixer, monoToStereo } from "./mixer.ts";
import type { Frame, Receiver } from "./receiver.ts";
import type { Player } from "./player.ts";

const TICK_MS = 10;
// Если таймер отстал сильнее (процесс подвис), догонять не пытаемся — перескакиваем.
const MAX_LAG_MS = 200;

export class Echo {
  readonly #mixer: Mixer;
  readonly #player: Player;
  readonly #onFrame = (frame: Frame) => this.#mixer.push(frame.userId, frame.pcm, frame.t);
  #timer: ReturnType<typeof setInterval> | null = null;
  #next = 0;

  constructor(receiver: Receiver, player: Player, delayMs: number) {
    this.#mixer = new Mixer(delayMs);
    this.#player = player;
    receiver.on("frame", this.#onFrame);
  }

  start(): void {
    if (this.#timer) return;
    this.#next = Date.now();
    this.#timer = setInterval(() => this.#step(), TICK_MS);
  }

  stop(): void {
    if (this.#timer) clearInterval(this.#timer);
    this.#timer = null;
    this.#mixer.clear();
    this.#player.stop();
  }

  // Кадры выдаются по виртуальным часам с шагом 20 мс, а не по срабатываниям таймера,
  // поэтому неточность setInterval не копится.
  #step(): void {
    const now = Date.now();
    if (now - this.#next > MAX_LAG_MS) this.#next = now;
    while (this.#next <= now) {
      const mixed = this.#mixer.pull(this.#next);
      if (mixed) this.#player.write(monoToStereo(mixed));
      this.#next += FRAME_MS;
    }
  }
}
