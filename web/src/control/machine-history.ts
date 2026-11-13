// The last minutes of the machine (each GPU, its memory and temperature, CPU, RAM), one point per status (every 4 s),
// kept in the page for the server's panel of the top bar (TopMachine.vue).
import {shallowRef} from 'vue';
import type {Gpu, SystemInfo} from '../common/panel-types';

/** 90 points: 6 minutes. */
export const HISTORY_POINTS = 90;

/** The points of one GPU: its use, its memory in use (0-100) and its temperature. */
export interface CardHistory {
  use: number[];
  memory: number[];
  temp: number[];
}

export interface MachineHistory {
  /** One per card, in nvidia-smi's order. */
  gpus: CardHistory[];
  cpu: number[];
  ram: number[];
}

export const machineHistory = shallowRef<MachineHistory>({gpus: [], cpu: [], ram: []});
/** The machine at the latest status, null before the first. */
export const machineNow = shallowRef<SystemInfo | null>(null);

function push(list: number[], value: number | null): number[] {
  return value === null ? list : [...list, value].slice(-HISTORY_POINTS);
}

/** The GPUs of the machine, each on its own: system.gpus from a server that reads each card (#29), else the one gpu of
 * an older server; none on a machine without nvidia-smi. */
export function gpuCards(system: SystemInfo | null | undefined): Gpu[] {
  if (!system) return [];
  return system.gpus?.length ? system.gpus : system.gpu ? [system.gpu] : [];
}

/** The cards the server's panel shows: this machine's (gpuCards), else those of CT 131 its GPU jobs run on. */
export function shownCards(system: SystemInfo | null | undefined): Gpu[] {
  const here = gpuCards(system);
  return here.length ? here : system?.remote_gpus?.cards ?? [];
}

/** A card by its number, "GPU 0", "GPU 1", when the machine has more than one; "GPU" when it has one. position: its
 * place in gpuCards, for a card without an index (the gpu of an older server). */
export function cardLabel(card: Gpu, position: number, count: number): string {
  return count > 1 ? 'GPU ' + (card.index ?? position) : 'GPU';
}

/** Its PCI bus as nvidia-smi prints it, "81:00.0", "C1:00.0"; '' when the server does not say. */
export const cardBus = (card: Gpu): string => (card.bus_id ?? '').toUpperCase();

/** The memory of a card in use, 0-100, or null when unknown. */
export function gpuMemoryPercent(card: Gpu): number | null {
  return card.memory_total_mb && card.memory_used_mb !== null ? card.memory_used_mb * 100 / card.memory_total_mb : null;
}

/** "29.4 of 32 GB": the memory of a card in use; '' when unknown. */
export function cardMemoryText(card: Gpu): string {
  return card.memory_used_mb === null || !card.memory_total_mb ? ''
    : (card.memory_used_mb / 1024).toFixed(1) + ' of ' + Math.round(card.memory_total_mb / 1024) + ' GB';
}

/** The card with the most memory free (where Qwen3-VL and a preparation would load), null without a GPU. */
export function freestCard(system: SystemInfo | null | undefined): Gpu | null {
  const room = (card: Gpu): number => (card.memory_total_mb ?? 0) - (card.memory_used_mb ?? 0);
  return gpuCards(system).reduce<Gpu | null>((best, card) => best === null || room(card) > room(best) ? card : best, null);
}

/** One more point, from a status. */
export function recordMachine(system: SystemInfo): void {
  const before = machineHistory.value;
  machineNow.value = system;
  machineHistory.value = {
    gpus: shownCards(system).map((card, position) => {
      const was = before.gpus[position] ?? {use: [], memory: [], temp: []};
      return {use: push(was.use, card.utilization ?? 0), memory: push(was.memory, gpuMemoryPercent(card)),
        temp: push(was.temp, card.temperature)};
    }),
    cpu: push(before.cpu, system.cpu ? system.cpu.total : null),
    ram: push(before.ram, system.memory ? system.memory.used_percent : null),
  };
}

export interface MachineCell {
  key: string;
  /** What it measures, for its name and the colour of its bar. */
  kind: 'gpu' | 'gpu-memory' | 'gpu-temp' | 'cpu' | 'ram' | 'disk';
  label: string;
  value: string;
  /** The tooltip; the small line under its name in the server's panel. */
  title: string;
  /** The last points, 0-100 (the temperature in °C), for the sparkline; empty for the disk. */
  values: number[];
  color: string;
  /** A hot GPU (80 °C and more). */
  hot?: boolean;
  /** How full it is now, 0-100: what the 5-notch stack of the tab row shows (the temperature from 30 to 90 °C). */
  level: number;
  /** A GPU of a machine with more than one: the card (what holds it), and its name in the panel, "GPU 0 · 81:00.0". */
  card?: Gpu;
  name?: string;
  /** The cards of CT 131 when it does not answer: a row that says so, out of the pill's GPU load. */
  off?: boolean;
}

/** One GPU: its use, memory and temperature as three gauges. */
function cardCells(card: Gpu, history: CardHistory | undefined, recent: (values: number[]) => number[], short: boolean): MachineCell[] {
  const list: MachineCell[] = [{key: 'gpu', kind: 'gpu', label: 'GPU', value: Math.round(card.utilization ?? 0) + '%',
    title: card.name, values: recent(history?.use ?? []), color: 'var(--spark-gpu)', level: card.utilization ?? 0}];
  const used = gpuMemoryPercent(card);
  if (used !== null && card.memory_used_mb !== null && card.memory_total_mb) {
    list.push({key: 'gpu-memory', kind: 'gpu-memory', label: short ? 'VRAM' : 'GPU memory',
      value: (card.memory_used_mb / 1024).toFixed(1) + ' GB', title: Math.round(used) + '% of ' + Math.round(card.memory_total_mb / 1024) + ' GB',
      values: recent(history?.memory ?? []), color: 'var(--spark-gpu)', level: used});
  }
  if (card.temperature !== null) {
    list.push({key: 'gpu-temp', kind: 'gpu-temp', label: short ? 'Temp' : 'GPU temp', value: Math.round(card.temperature) + ' °C',
      title: card.power_w === null ? card.name : 'Drawing ' + Math.round(card.power_w) + ' W', values: recent(history?.temp ?? []),
      color: 'var(--spark-gpu)', hot: card.temperature >= 80, level: (card.temperature - 30) * 100 / 60});
  }
  return list;
}

/** One GPU of several in one gauge: its use, with its memory and temperature in its small line ("29.4 of 32 GB · 64 °C"),
 * so two cards stay short in the panel. */
function cardCell(card: Gpu, position: number, count: number, history: CardHistory | undefined,
                  recent: (values: number[]) => number[], host = ''): MachineCell {
  const label = (host ? host + ' · ' : '') + cardLabel(card, position, count);
  const bus = cardBus(card);
  const title = [cardMemoryText(card), card.temperature === null ? '' : Math.round(card.temperature) + ' °C']
    .filter(Boolean).join(' · ');
  return {key: 'gpu-' + (card.index ?? position), kind: 'gpu', label, name: bus ? label + ' · ' + bus : label,
    value: Math.round(card.utilization ?? 0) + '%', title: title || card.name, values: recent(history?.use ?? []),
    color: 'var(--spark-gpu)', hot: card.temperature !== null && card.temperature >= 80, level: card.utilization ?? 0, card};
}

/** GPU, GPU memory and temperature, CPU, RAM and (with disk) the free disk: each its value now and its last points; a
 * machine with several GPUs has one gauge per card instead (cardCell). short: the short labels of the tab row (VRAM,
 * Temp). */
export function machineCells(system: SystemInfo, points: number, disk = false, short = false): MachineCell[] {
  const {cpu, memory} = system;
  const history = machineHistory.value;
  const recent = (values: number[]): number[] => values.slice(-points);
  const cards = gpuCards(system);
  const list: MachineCell[] = cards.length === 1 && cards[0] ? cardCells(cards[0], history.gpus[0], recent, short)
    : cards.map((card, position) => cardCell(card, position, cards.length, history.gpus[position], recent));
  // No card here: those of CT 131, where the GPU runner sends the GPU jobs (#77), each as one of several.
  const remote = cards.length ? null : system.remote_gpus;
  if (remote?.cards.length) {
    list.push(...remote.cards.map((card, position) => cardCell(card, position, Math.max(2, remote.cards.length),
      history.gpus[position], recent, remote.host)));
  } else if (remote && !remote.reachable) {
    list.push({key: 'gpu-remote', kind: 'gpu', label: remote.host + ' GPUs', name: remote.host + ' · V100s', value: 'Off',
      title: 'Not answering: Cristian or cluster-admin start it', values: [], color: 'var(--spark-gpu)', level: 0, off: true});
  }
  if (cpu) {
    list.push({key: 'cpu', kind: 'cpu', label: 'CPU', value: Math.round(cpu.total) + '%', title: 'all cores',
      values: recent(history.cpu), color: 'var(--spark-cpu)', level: cpu.total});
  }
  if (memory) {
    list.push({key: 'ram', kind: 'ram', label: 'RAM', value: Math.round(memory.used_percent) + '%',
      title: (memory.total_gb - memory.available_gb).toFixed(1) + ' of ' + memory.total_gb + ' GB in use',
      values: recent(history.ram), color: 'var(--spark-cpu)', level: memory.used_percent});
  }
  if (disk && system.disk) {
    const used = 100 - system.disk.free_gb * 100 / system.disk.total_gb;
    list.push({key: 'disk', kind: 'disk', label: 'Disk', value: Math.round(system.disk.free_gb) + ' GB free',
      title: Math.round(used) + '% of ' + Math.round(system.disk.total_gb) + ' GB in use', values: [], color: '',
      level: used});
  }
  return list;
}
