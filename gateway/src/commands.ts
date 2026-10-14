// Slash-команды /join и /leave, проверка whitelist и автовыход из пустого канала.

import {
  type ChatInputCommandInteraction,
  type Guild,
  MessageFlags,
  SlashCommandBuilder,
} from "discord.js";

export const COMMANDS = [
  new SlashCommandBuilder().setName("join").setDescription("Позвать Клона в твой голосовой канал"),
  new SlashCommandBuilder().setName("leave").setDescription("Отпустить Клона из голосового канала"),
].map((c) => c.toJSON());

// Команды регистрируются на сервере, а не глобально: так они появляются сразу.
export async function registerCommands(guild: Guild): Promise<void> {
  await guild.commands.set(COMMANDS);
}

export const REPLIES = {
  denied: "Эта команда только для участников из whitelist.",
  notInVoice: "Зайди в голосовой канал, потом вызывай /join.",
  joined: (channel: string) => `Захожу в «${channel}».`,
  joinFailed: "Не получилось подключиться к каналу, подробности в логе шлюза.",
  left: "Выхожу.",
  notConnected: "Я и так не в голосовом канале.",
} as const;

export type CommandDecision =
  | { kind: "deny" }
  | { kind: "not-in-voice" }
  | { kind: "join"; channelId: string }
  | { kind: "leave" };

export interface CommandRequest {
  command: string;
  userId: string;
  // Голосовой канал, где сейчас сидит вызвавший, если он там есть.
  voiceChannelId: string | null;
}

// Проверка прав и выбор действия без Discord: её и тестируем.
export function decideCommand(req: CommandRequest, whitelist: readonly string[]): CommandDecision {
  if (!whitelist.includes(req.userId)) return { kind: "deny" };
  if (req.command === "leave") return { kind: "leave" };
  if (req.voiceChannelId === null) return { kind: "not-in-voice" };
  return { kind: "join", channelId: req.voiceChannelId };
}

export interface CommandActions {
  // Возвращает имя канала или бросает ошибку.
  join(channelId: string): Promise<string>;
  // false — Клон и так не был в канале.
  leave(): Promise<boolean>;
}

export async function handleCommand(
  interaction: ChatInputCommandInteraction,
  whitelist: readonly string[],
  actions: CommandActions,
): Promise<void> {
  const member = interaction.guild?.members.cache.get(interaction.user.id);
  const decision = decideCommand(
    {
      command: interaction.commandName,
      userId: interaction.user.id,
      voiceChannelId: member?.voice.channelId ?? null,
    },
    whitelist,
  );
  const reply = (content: string) => interaction.reply({ content, flags: MessageFlags.Ephemeral });

  switch (decision.kind) {
    case "deny":
      console.log(`[commands] /${interaction.commandName} от ${interaction.user.id}: не в whitelist, отказ`);
      return void (await reply(REPLIES.denied));
    case "not-in-voice":
      return void (await reply(REPLIES.notInVoice));
    case "leave":
      return void (await reply((await actions.leave()) ? REPLIES.left : REPLIES.notConnected));
    case "join": {
      // Подключение под DAVE может занять дольше 3 с, отведённых на ответ.
      await interaction.deferReply({ flags: MessageFlags.Ephemeral });
      try {
        const name = await actions.join(decision.channelId);
        await interaction.editReply(REPLIES.joined(name));
      } catch (e) {
        console.error(`[commands] /join: ${e instanceof Error ? e.message : String(e)}`);
        await interaction.editReply(REPLIES.joinFailed);
      }
    }
  }
}

// Автовыход: когда в канале не осталось людей, через delayMs вызывается onExpire.
// Если кто-то вернулся раньше — таймер отменяется.
export class AutoLeave {
  #timer: ReturnType<typeof setTimeout> | null = null;
  readonly #delayMs: number;
  readonly #onExpire: () => void;

  constructor(delayMs: number, onExpire: () => void) {
    this.#delayMs = delayMs;
    this.#onExpire = onExpire;
  }

  get pending(): boolean {
    return this.#timer !== null;
  }

  // Вызывать при каждом изменении состава канала (и сразу после входа).
  update(humans: number): void {
    if (humans > 0) {
      this.cancel();
    } else if (this.#timer === null) {
      this.#timer = setTimeout(() => {
        this.#timer = null;
        this.#onExpire();
      }, this.#delayMs);
    }
  }

  cancel(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
  }
}
