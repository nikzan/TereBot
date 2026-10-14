// Воспроизведение PCM в голосовой канал: AudioPlayer + Raw-ресурс (s16le 48 кГц стерео).
// Ресурс создаётся при первой записи и живёт, пока поток не иссякнет: тогда плеер сам уходит
// в Idle, и индикатор «говорит» у Клона гаснет. stop() обрывает звук сразу.

import { PassThrough } from "node:stream";

import {
  AudioPlayerStatus,
  NoSubscriberBehavior,
  StreamType,
  type VoiceConnection,
  createAudioPlayer,
  createAudioResource,
} from "@discordjs/voice";

import { logger } from "./log.ts";

const log = logger("player");

// Сколько кадров по 20 мс без данных плеер терпит, прежде чем остановиться.
// Покрывает дрожание таймера и короткие провалы приёма внутри фразы.
const MAX_MISSED_FRAMES = 25;

export class Player {
  readonly #player = createAudioPlayer({
    behaviors: { noSubscriber: NoSubscriberBehavior.Pause, maxMissedFrames: MAX_MISSED_FRAMES },
  });
  #input: PassThrough | null = null;

  constructor() {
    this.#player.on("stateChange", (_old, state) => {
      if (state.status === AudioPlayerStatus.Idle) this.#dropInput();
    });
    this.#player.on("error", (e) => log(`ошибка плеера: ${e.message}`));
  }

  attach(connection: VoiceConnection): void {
    connection.subscribe(this.#player);
  }

  // pcm — s16le 48 кГц стерео.
  write(pcm: Buffer): void {
    if (!this.#input) {
      this.#input = new PassThrough();
      this.#player.play(createAudioResource(this.#input, { inputType: StreamType.Raw }));
    }
    this.#input.write(pcm);
  }

  // Немедленная остановка: плеер сразу переходит в Idle, буфер выбрасывается.
  stop(): void {
    this.#player.stop(true);
    this.#dropInput();
  }

  #dropInput(): void {
    this.#input?.destroy();
    this.#input = null;
  }
}
