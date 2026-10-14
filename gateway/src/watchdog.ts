// Правила восстановления приёма: события приёмника + время → переподписка или переподключение.
// Чистая логика без Discord и таймеров: время передаётся аргументом, поэтому тестируется на фиктивных часах.
//
// 1. Поток подписки упал с ошибкой, пока участник говорит → немедленная переподписка.
// 2. Участник говорит (speaking start), а кадров от него нет дольше silenceMs → переподписка.
// Переподписка считается неудачной, пока от участника не пришёл кадр. Если у участника уже
// maxResubscribes неудачных переподписок за windowMs, вместо следующей — переподключение к каналу.

export type WatchdogAction =
  | { kind: "resubscribe"; userId: string; reason: string }
  | { kind: "reconnect"; userId: string; reason: string };

export interface WatchdogOptions {
  silenceMs: number;
  maxResubscribes: number;
  windowMs?: number;
}

interface UserState {
  speaking: boolean;
  // Последний момент, когда приём от участника «двигался»: начало речи, кадр или переподписка.
  lastProgress: number;
  // Время переподписок, после которых кадров ещё не было.
  failed: number[];
}

export class Watchdog {
  readonly #silenceMs: number;
  readonly #maxResubscribes: number;
  readonly #windowMs: number;
  #users = new Map<string, UserState>();

  constructor(options: WatchdogOptions) {
    this.#silenceMs = options.silenceMs;
    this.#maxResubscribes = options.maxResubscribes;
    this.#windowMs = options.windowMs ?? 60_000;
  }

  #user(userId: string, t: number): UserState {
    let user = this.#users.get(userId);
    if (!user) {
      user = { speaking: false, lastProgress: t, failed: [] };
      this.#users.set(userId, user);
    }
    return user;
  }

  speakingStart(userId: string, t: number): void {
    const user = this.#user(userId, t);
    user.speaking = true;
    user.lastProgress = t;
  }

  speakingEnd(userId: string, t: number): void {
    this.#user(userId, t).speaking = false;
  }

  frame(userId: string, t: number): void {
    const user = this.#user(userId, t);
    user.lastProgress = t;
    user.failed = [];
  }

  streamError(userId: string, t: number): WatchdogAction[] {
    const user = this.#user(userId, t);
    if (!user.speaking) return [];
    return [this.#trigger(userId, user, t, "поток подписки упал с ошибкой")];
  }

  tick(t: number): WatchdogAction[] {
    const actions: WatchdogAction[] = [];
    for (const [userId, user] of this.#users) {
      if (user.speaking && t - user.lastProgress >= this.#silenceMs) {
        const action = this.#trigger(userId, user, t, `говорит, но кадров нет ${t - user.lastProgress} мс`);
        actions.push(action);
        // Переподключение пересоздаёт все подписки — остальных в этом такте не трогаем.
        if (action.kind === "reconnect") break;
      }
    }
    return actions;
  }

  // Соединение пересоздано: вся история недействительна.
  reset(): void {
    this.#users.clear();
  }

  #trigger(userId: string, user: UserState, t: number, reason: string): WatchdogAction {
    user.failed = user.failed.filter((at) => t - at < this.#windowMs);
    if (user.failed.length >= this.#maxResubscribes) {
      this.reset();
      return {
        kind: "reconnect",
        userId,
        reason: `${reason}; ${user.failed.length} неудачных переподписок за ${this.#windowMs / 1000} с`,
      };
    }
    user.failed.push(t);
    user.lastProgress = t;
    return { kind: "resubscribe", userId, reason };
  }
}
