// Строки лога шлюза: время ISO, тег модуля, сообщение. Разбор Эхо-теста опирается на этот формат.

export type Log = (message: string) => void;

export function logger(tag: string): Log {
  return (message) => console.log(`${new Date().toISOString()} [${tag}] ${message}`);
}
