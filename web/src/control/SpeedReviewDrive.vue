<script setup lang="ts">
// Data › Speed review, a drive opened ("/speed-review?drive=<route>"): the drive player (DriveReviewDrive.vue)
// with one lane more, the answer the brain learns, and the commands at its side. The answer is his own dashboard speed
// at every moment (speed_review.py driver_speed): nothing to propose and nothing to confirm. A block is the only
// correction: "Block from here" keeps a start on the page, "Block to here" saves the block from that start to the time
// shown with its kph, which starts from his speed at the start, or left out of the training. A block is changed by saving
// it again with the same ends, and removed with its start. The lane draws his speed as a step line, the blocks over it with
// their kph, the roadworks and how far the drive was gone through for them. Roadworks are marked next to the speed, two
// taps on the frame shown; a third tap says that the drive was gone through up to the frame shown, so that no mark there
// means no works, and the drive opens there again. The kinds of road (narrow, country, gravel, potholes, and those added
// with "+") are marked the same way, each with its own marks and its own "checked to here": a tap on a chip starts a mark of
// it at the frame shown and a second tap ends it there; the chip is solid where the frame shown is inside one of its
// marks, and the last one tapped is the one the lane draws and Remove / Checked to here work on. The road type
// (town, rural, motorway…) is as the brain takes it, one a second: the map's, the road labeler's over it where it is sure
// (Prepare Speed review, definitive, nothing to confirm), his correction over both. It is drawn on a lane of its own,
// light where it is the map's, a mid tone where it is the labeler's (net), solid where he corrected it, and the
// lane's name and the tag say whose the stretch shown is (map / net / yours). He corrects only where it is wrong:
// a chip sets a type, or "can't tell", on the whole stretch of the frame shown, on its part before or after the frame
// shown (✂ Before / After, the stretch cut there), or from a frame kept with "From here" to the frame shown
// (speed_review.save_road_type), kept as a correction over both layers (road_types.correct). The picture
// shows the answer (its square named ANSWER, as the lane) next to his dashboard speed (the car's square too: never the car's vEgo, which reads lower), so that the number read
// is the right one. Where openpilot or the car's cruise held the speed (held) it is not his foot: the lane greys it and
// ANSWER is a dash, as the training leaves it out, unless a block of his says a speed there. A tap on a limit of the map's lane makes a block with it, from its start to its end, to change or
// remove as any other. The kph of a block is stepped or typed; its ends are dragged on the lane, pulled as by a magnet onto
// where the map's limit changes, the ends of the other blocks and the time shown, and the block is moved in one write.
// Every command has its key, listed under the card on a screen with a keyboard.
// The signs of Sign review are on it too (sign-frames.ts): on the picture the boxes of the very frame the video presents,
// 20 a second (sign-track.ts: the networks' boxes on every frame, a frame of Sign review confirmed winning on its own frame;
// without them Sign review's frames only, one a second), solid where the frame is confirmed, dashed the proposals; a lane of the confirmed speed signs, a tap on
// one going there; and at the top of the side a switch Speed | Signs that changes only the side. Signs is Sign review's
// editor (review/embed.ts): paused on a frame, its boxes are moved, resized, drawn and deleted on the picture, its chips
// and Confirm are at the side, saved as on the page of Sign review, and its keys are Sign review's (← → a frame, ⇧ ← → a
// second). The Speed side keeps the speed's keys. Without the editor's data (an older server) the Signs side lists the
// boxes read only; without the signs (a drive without frames in Sign review, a guest given only its speed) the page is the
// same, without them. This side is where the signs are reviewed since the page of Sign review was retired (9 Nov 2026):
// a link of the signs (common/sign-link.ts, "?side=signs&frame=<key>&segment=N&second=S", the server's address of Sign
// review leading there) opens it on that frame, else the frame nearest the moment, else the first still to review; there
// too "Review with AI" (components/AiRouteSheet.vue). A guest given only the drive's signs has only this side. At the top
// of the Speed side, the drive's switch "OK for brain training" (SpeedTrainingSwitch.vue): only the drives with it on
// enter the brain training, each whole.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, ApiError, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {isGuest} from '../common/guest';
import {askedSigns} from '../common/sign-link';
import AiRouteSheet from '../components/AiRouteSheet.vue';
import {toast} from '../composables/toast';
import {boxColor, describe, shortName} from '../review/catalog';
import {row as editRow} from '../review/current';
import {frameKeyAt, loadSignEditor, stopSignEditor, videoFrameOf} from '../review/embed';
import {frameStatus} from '../review/frames';
import {linkedFrame, openFrame} from '../review/navigation';
import SignEditorPanel from '../review/SignEditorPanel.vue';
import SignEditorStage from '../review/SignEditorStage.vue';
import {report as editReport, rowOf, state as editState} from '../review/store';
import {clockOf, type ScanState, secondOf as frameSecondOf, type SyncSegment} from './drive-review';
import DriveReviewDrive from './DriveReviewDrive.vue';
import Segmented from './Segmented.vue';
import SpeedTrainingSwitch from './SpeedTrainingSwitch.vue';
import {isOtherRoad, isSpeedSign, nearestFrame, placeFrames, type SignFrames, signRuns} from './sign-frames';
import {boxesAt, type Drawn, exactFrames, reviewedFrames, type SignTrack, trackFrames, type VideoFrame, type VideoIndex} from './sign-track';
import {type Block, blockAt, canEndWorks, clampKph, draggedEdge, heldAt, MIN_BLOCK_NS, ROAD_TYPE_LABELS, ROADWORKS, type RoadKind,
  roadTypeAt, roadTypeLabel, type RoadTypeSource, type RoadTypeStretch, type SpeedDrive, speedAt, speedIndex, typeSource, type Works, worksAt, type WorksTap} from './speed-review';

/** kinds: a guest's work on the drive ('speed', 'signs'), which says the sides it has; the administrator has both. */
const props = defineProps<{route: string; readOnly: boolean; kinds?: string[] | null}>();
const RATES = [1, 2, 4] as const;
const HEIGHT = 84;
/** The speed is drawn from TOP to PLOT; under it the bar of the marks picked and the line of what was gone through. */
const TOP = 16;
const PLOT = 74;
/** The lane of the road types. */
const TYPE_HEIGHT = 18;
/** The lane of the confirmed speed signs. */
const SIGN_HEIGHT = 24;
/** A new block with no speed of his at its start begins at this kph. */
const FALLBACK_KPH = 50;

const player = useTemplateRef<InstanceType<typeof DriveReviewDrive>>('player');
const seconds = ref(0);
const t0 = ref(0);
function follow(at: number, start: number): void {
  seconds.value = at;
  t0.value = start;
}
const now = computed(() => Math.round(t0.value + seconds.value * 1e9));
const secondOf = (ns: number): number => (ns - t0.value) / 1e9;

// -- the drive -------------------------------------------------------------------------------------------------------------
const drive = shallowRef<SpeedDrive | null>(null);
const error = ref('');
const address = (): string => '/api/speed-review/drives/' + encodeURIComponent(props.route);
function show(answer: SpeedDrive): void {
  // A cut answer (the page left while it came) reads as {message} in api(): never a drive without its blocks.
  if (!answer || !Array.isArray(answer.blocks) || !Array.isArray(answer.speed)) throw new Error('The server sent a drive this page cannot read.');
  drive.value = answer;
}
async function read(): Promise<void> {
  try {
    show(await api<SpeedDrive>(address()));
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
/** Before its logs are read, what keeps the speed away: the search at work with its share, or what to do. */
const noSpeed = computed((): string => {
  const scan = drive.value?.scan;
  if (scan?.state === 'queued' || scan?.state === 'scanning') {
    return 'No speed to show yet: the logs of this drive are being read' + (scan.state === 'scanning' ? ', '
      + Math.round((scan.progress || 0) * 100) + '%' : '') + '. The speed shows here when they are.';
  }
  if (scan?.state === 'failed') return 'No speed to show: reading the logs failed (' + (scan.error ?? 'unknown error') + ').';
  if (scan?.state === 'no_logs') return 'No speed to show: this drive has no logs on the server.';
  return 'No speed to show: the logs of this drive are not read yet. Search the logs, at the top, reads them.';
});
/** The player's search of the logs: its share while it runs, and the drive again when it ends. */
function onScan(state: ScanState): void {
  if (!drive.value) return;
  if (state.state === 'queued' || state.state === 'scanning') {
    drive.value = {...drive.value, scan: {state: state.state, progress: state.progress, error: state.error}};
  } else if (drive.value.scan?.state === 'queued' || drive.value.scan?.state === 'scanning'
    || (!drive.value.span_ns && ['done', 'no_episodes', 'outdated'].includes(state.state))) {
    void read();
  }
}
const blocks = computed(() => drive.value?.blocks ?? []);
const speed = computed(() => drive.value?.speed ?? []);
const held = computed(() => drive.value?.held ?? []);
const maxKph = computed(() => drive.value?.max_kph ?? 200);
/** His dashboard speed at the time shown, and the block there: the answer is the block's when there is one. */
const driverKph = computed(() => speedAt(speed.value, now.value));
const block = computed(() => blockAt(blocks.value, now.value));
/** openpilot or the car's cruise held the speed here, and no block says one: no answer. */
const machine = computed(() => !block.value && heldAt(held.value, now.value));
const label = computed(() => block.value ? block.value.kph : machine.value ? null : driverKph.value);

const go = (item: Block): void => {
  // A millisecond in: the time shown is on this block, not at the end of the one before.
  player.value?.seek(secondOf(item.start_ns) + 0.001);
};
// A drive opens where what is picked (the roadworks, or a kind of road) was gone through to, the review going on where it
// stopped, else at its start. Opened at a moment ("?segment=N&second=S") or on the signs ("?side=signs"), it stays
// there: the player seeks to it, or the Signs side to its frame.
const asked = askedSigns();
const linked = new URLSearchParams(location.search).has('segment') || asked.side || asked.frame !== null;
const toStart = watch([t0, drive], () => {
  if (!t0.value || !drive.value) return;
  toStart();
  if (!linked && checked.value !== null) player.value?.seek(secondOf(checked.value));
});

// -- the commands ----------------------------------------------------------------------------------------------------------
const saving = ref(false);
const locked = computed(() => saving.value || props.readOnly || signsOnly);
async function send(path: string, body: object): Promise<boolean> {
  if (locked.value || !drive.value) return false;
  saving.value = true;
  try {
    show(await post<SpeedDrive>(address() + path, {...body, base_revision: drive.value.revision}));
    return true;
  } catch (problem) {
    toast(messageOf(problem), true);
    // A stale revision or anything else: what the server has now.
    void read();
    return false;
  } finally {
    saving.value = false;
  }
}

// A new block: its start stays on the page until its end is given; then it is saved at once. Its ends are in the
// drive, and an end before the start swaps them: the block goes from the earlier time to the later one.
const pending = ref<number | null>(null);
const pendingKph = ref(FALLBACK_KPH);
const pendingOut = ref(false);
const within = (ns: number): number => {
  const span = drive.value?.span_ns;
  return span ? Math.min(span[1], Math.max(span[0], ns)) : ns;
};
const canStart = computed(() => !props.readOnly && !!drive.value?.span_ns);
function startBlock(): void {
  if (!canStart.value) return;
  pending.value = within(now.value);
  pendingKph.value = clampKph(speedAt(speed.value, pending.value) ?? FALLBACK_KPH, maxKph.value);
  pendingOut.value = false;
}
const ends = computed((): [number, number] | null => pending.value === null ? null
  : [within(Math.min(pending.value, now.value)), within(Math.max(pending.value, now.value))]);
const canEnd = computed(() => !!ends.value && ends.value[1] - ends.value[0] >= MIN_BLOCK_NS);
async function endBlock(): Promise<void> {
  if (!ends.value || !canEnd.value) return;
  const [start, end] = ends.value;
  if (!await send('/blocks', {start_ns: start, end_ns: end, kph: pendingOut.value ? null : pendingKph.value, note: ''})) return;
  pending.value = null;
  // A block ends just before its end: a millisecond back, the card shows the block just made, ready to change.
  if (now.value >= end) player.value?.seek(secondOf(end) - 0.001);
}
const cancelBlock = (): void => { pending.value = null; };
/** A tap on a limit of the map's lane: a block with that limit over the whole stretch the map had it, saved at once. */
async function fromMap(from: number, to: number, kph: number): Promise<void> {
  const start = within(Math.round(t0.value + from * 1e9));
  const end = within(Math.round(t0.value + to * 1e9));
  if (end - start < MIN_BLOCK_NS) {
    toast('This limit of the map lasts less than 1 s: no block.');
    return;
  }
  const speed = clampKph(kph, maxKph.value);
  if (await send('/blocks', {start_ns: start, end_ns: end, kph: speed, note: 'map'})) {
    toast('A block of ' + speed + ' km/h from the map, ' + clockOf(secondOf(start)) + '–' + clockOf(secondOf(end)) + '.');
  }
}
/** Key m: the block of the map's limit at the time shown, as a tap on it. */
function mapBlock(): void {
  if (!canStart.value) return;
  const run = player.value?.mapRunAt(seconds.value) ?? null;
  if (run) void fromMap(run.from, run.to, run.kph);
  else toast('The map has no limit here.');
}
function tapBlock(): void {
  if (pending.value === null) startBlock();
  else if (canEnd.value) void endBlock();
  else toast('A block lasts at least 1 s: go on to its end first.');
}

// A saved block is changed by saving it again with its ends and its note; Remove gives his speed back there.
const setBlock = (item: Block, kph: number | null): void => {
  void send('/blocks', {start_ns: item.start_ns, end_ns: item.end_ns, kph, note: item.note});
};
const removeBlock = (item: Block): void => { void send('/blocks', {remove: true, start_ns: item.start_ns}); };
/** Key l: the new block, else the block at the time shown, left out of the training, or given a speed again. */
function leaveOut(): void {
  if (pending.value !== null) pendingOut.value = !pendingOut.value;
  else if (block.value) setBlock(block.value, block.value.kph === null ? kphOf(block.value) : null);
}
/** Key x: the block at the time shown removed, his speed back there. */
function removeHere(): void {
  if (block.value) removeBlock(block.value);
  else toast('No block here to remove.');
}
/** A step of the kph of the new block, else of the block at the time shown. */
function step(delta: number): void {
  if (pending.value !== null) {
    if (!pendingOut.value) pendingKph.value = clampKph(pendingKph.value + delta, maxKph.value);
  } else if (block.value && block.value.kph !== null) {
    const kph = clampKph(block.value.kph + delta, maxKph.value);
    if (kph !== block.value.kph) setBlock(block.value, kph);
  }
}
// A kph typed in its field: kept as typed while the field is used (the card is drawn again at every frame), then given to
// the block it was typed for, found by its start: the new block ('new'), or a saved one, whatever the time shown is by
// then. Focusing the field stops the video. Only a whole number of the scale is taken; anything else is said and dropped.
const draft = shallowRef<{key: number | 'new'; text: string; bad: boolean} | null>(null);
const draftOf = (key: number | 'new'): string | null => draft.value?.key === key ? draft.value.text : null;
function typing(key: number | 'new', event: Event): void {
  const input = event.target as HTMLInputElement;
  draft.value = {key, text: input.value, bad: input.validity.badInput};
}
async function typed(): Promise<void> {
  const typed = draft.value;
  if (!typed) return;
  const key = typed.key;
  const kph = Number(typed.text);
  const good = !typed.bad && typed.text.trim() !== '' && Number.isInteger(kph) && kph >= 1 && kph <= maxKph.value;
  if (!good) {
    draft.value = null;
    if (typed.bad || typed.text.trim() !== '') toast('A speed is a whole number from 1 to ' + maxKph.value + ' km/h.', true);
    return;
  }
  if (key === 'new') {
    if (pending.value !== null && !pendingOut.value) pendingKph.value = kph;
    draft.value = null;
    return;
  }
  const item = blocks.value.find((other) => other.start_ns === key);
  // Shown as typed until the server answers: the old kph does not come back meanwhile.
  if (item && item.kph !== null && kph !== item.kph) await send('/blocks', {start_ns: item.start_ns, end_ns: item.end_ns, kph, note: item.note});
  if (draft.value === typed) draft.value = null;
}
/** The field left without a change: what was typed goes. */
function untyped(): void {
  if (!saving.value) draft.value = null;
}
const kphOf = (item: Block): number => speedAt(speed.value, item.start_ns) ?? FALLBACK_KPH;
const blockText = (item: Block): string => item.kph === null ? 'left out' : item.kph + ' km/h';

// -- the roadworks and the kinds of road -----------------------------------------------------------------------------------
/** The roadworks first, then the kinds of road: the one picked is what the taps mark. */
const kinds = computed((): RoadKind[] => [{kind: ROADWORKS, label: 'Roadworks'}, ...(drive.value?.road_kinds ?? [])]);
const PICKED_KEY = 'speed-review-kind';
function readPicked(): string {
  try {
    return localStorage.getItem(PICKED_KEY) ?? ROADWORKS;
  } catch {
    return ROADWORKS;
  }
}
const picked = ref(readPicked());
watch(picked, (kind) => {
  try {
    localStorage.setItem(PICKED_KEY, kind);
  } catch {
    // A private window keeps nothing: the roadworks next time.
  }
});
/** What is picked; a kind this drive's server does not have (an old choice) is the roadworks. */
const pickedKind = computed(() => kinds.value.find((item) => item.kind === picked.value) ?? kinds.value[0]!);
const isWorks = computed(() => pickedKind.value.kind === ROADWORKS);
const marksOf = (kind: string): Works[] => kind === ROADWORKS ? drive.value?.works ?? []
  : (drive.value?.road ?? []).filter((item) => item.kind === kind);
/** Its own key only: a kind is never read off what every object has. */
const checkedOf = (kind: string): number | null => {
  if (kind === ROADWORKS) return drive.value?.works_checked_ns ?? null;
  const checked = drive.value?.road_checked_ns ?? {};
  return Object.hasOwn(checked, kind) ? checked[kind] ?? null : null;
};
const marks = computed(() => marksOf(pickedKind.value.kind));
const markOpen = computed(() => marks.value.find((item) => item.end_ns === null) ?? null);
const markHere = computed(() => worksAt(marks.value, now.value));
const markShown = computed(() => markHere.value ?? markOpen.value);
const canMark = computed(() => markOpen.value ? canEndWorks(marks.value, markOpen.value, now.value) : markHere.value === null);
const checked = computed(() => checkedOf(pickedKind.value.kind));
const canCheck = computed(() => !markOpen.value || now.value < markOpen.value.start_ns);
/** The kinds whose marks the frame shown is inside: a dot on their chips. */
const inside = computed(() => new Set(kinds.value.filter((item) => worksAt(marksOf(item.kind), now.value)).map((item) => item.kind)));
const tapMark = (tap: WorksTap): void => {
  void (isWorks.value ? send('/works', {tap, at_ns: now.value})
    : send('/road', {kind: pickedKind.value.kind, tap, at_ns: now.value}));
};
function mark(): void {
  if (canMark.value) tapMark(markOpen.value ? 'end' : 'start');
}
/** A tap on a kind's chip: picked, and a mark of it started here, or its open one ended here; inside a mark it is only
 * picked (Remove this mark is under the chips). */
function tapKind(kind: string): void {
  picked.value = kind;
  if (locked.value) return;
  if (markOpen.value) {
    if (canMark.value) tapMark('end');
    else toast('Its end goes after its start, ' + clockOf(secondOf(markOpen.value.start_ns)) + '.', true);
  } else if (!markHere.value) {
    tapMark('start');
  }
}
/** Key k: the next kind. */
function nextKind(): void {
  const index = kinds.value.findIndex((item) => item.kind === pickedKind.value.kind);
  picked.value = kinds.value[(index + 1) % kinds.value.length]!.kind;
}
// A new kind of road from its name, saved at once and picked. A guest marks with the kinds there are and adds none.
const canAdd = computed(() => !props.readOnly && !isGuest());
const adding = ref(false);
const newLabel = ref('');
async function addKind(): Promise<void> {
  const label = newLabel.value.trim();
  if (!label || saving.value || !drive.value) return;
  saving.value = true;
  try {
    const answer = await post<{kind: string; road_kinds: RoadKind[]}>('/api/speed-review/road-kinds', {label});
    if (drive.value) drive.value = {...drive.value, road_kinds: answer.road_kinds};
    picked.value = answer.kind;
    newLabel.value = '';
    adding.value = false;
    toast('“' + label + '” added: mark it as the roadworks.');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}

// -- the road types --------------------------------------------------------------------------------------------------------
// A guest sees them and sets none: what comes back from a guest's copy is the speed review, not the map's corrections.
const roadTypes = computed(() => drive.value?.road_types ?? []);
/** The chips: the types in their order, then "can't tell" (null). */
const typeChoices = computed((): (string | null)[] => [...drive.value?.road_type_names ?? Object.keys(ROAD_TYPE_LABELS), null]);
/** The stretch at the frame shown, else one less than a second away: the map's first frame comes a moment after the
 * logs start (0:00 had none) and its last one a moment before they end. */
const typeHere = computed(() => roadTypeAt(roadTypes.value, now.value) ?? roadTypes.value.find((item) =>
  item.from_ns - 1e9 <= now.value && now.value < item.to_ns + 1e9) ?? null);
/** A stretch's type in words: where the map has none and nobody set one, it says so, not "can't tell". */
const typeName = (item: RoadTypeStretch): string => item.road_type === null && typeSource(item) !== 'person'
  ? 'None on the map' : roadTypeLabel(item.road_type);
/** Whose the stretch is, in a word: the lane's name and the tag. */
const TYPE_SOURCE_NAMES: Record<RoadTypeSource, string> = {person: 'yours', net: 'net', map: 'map'};
const TYPE_SOURCE_TAGS: Record<RoadTypeSource, string> = {person: 'Yours', net: 'Net', map: 'Map'};
const typeSourceName = (item: RoadTypeStretch | null): string => item ? TYPE_SOURCE_NAMES[typeSource(item)] : 'map';
const canType = computed(() => !props.readOnly && !isGuest() && roadTypes.value.length > 0);
/** A start kept with "From here": the next chip sets its type from there to the frame shown, not on the stretch. */
const typeFrom = ref<number | null>(null);
/** Whether a chip is the type in force at the frame shown: "can't tell" only where he said it, not where the map has none. */
const typeOn = (type: string | null): boolean => !!typeHere.value && typeHere.value.road_type === type
  && (type !== null || typeSource(typeHere.value) === 'person');
/** Which part of the stretch shown a chip sets: all of it, its start up to the frame shown, or from it to its end, so a
 * stretch is cut where the picture shows the change. */
type TypePart = 'all' | 'before' | 'after';
const typePart = ref<TypePart>('all');
const TYPE_PARTS: {key: TypePart; label: string}[] = [{key: 'all', label: 'Whole stretch'}, {key: 'before', label: '✂ Before'},
  {key: 'after', label: '✂ After'}];
// Another stretch: the whole of it again.
watch(() => typeHere.value?.from_ns, () => { typePart.value = 'all'; });
/** The stretch can be cut when the frame shown is inside it, half a second from its ends, and no "From here" is kept. */
const typeCuttable = computed(() => canType.value && typeFrom.value === null && !!typeHere.value
  && now.value > typeHere.value.from_ns + 5e8 && now.value < typeHere.value.to_ns - 5e8);
const typeCut = computed((): TypePart => typeCuttable.value ? typePart.value : 'all');
async function setType(type: string | null): Promise<void> {
  if (!canType.value) return;
  const from = typeFrom.value;
  const here = typeHere.value;
  if (from === null && !here) {
    toast('The map has no frame here: “From here” sets a type from one frame to another.');
    return;
  }
  // On the stretch: a frame shown just before its start or after its end sets it all the same.
  const at = here && from === null ? Math.min(here.to_ns - 1, Math.max(here.from_ns, within(now.value))) : within(now.value);
  // A part of the stretch: its ends are the frames of the map at those times, the frame shown in both parts.
  const body = from !== null ? {road_type: type, from_ns: from, to_ns: at}
    : here && typeCut.value === 'before' ? {road_type: type, from_ns: here.from_ns, to_ns: at}
    : here && typeCut.value === 'after' ? {road_type: type, from_ns: at, to_ns: here.to_ns - 1}
    : {road_type: type, at_ns: at};
  if (await send('/road-type', body)) {
    typeFrom.value = null;
    typePart.value = 'all';
  }
}
function startType(): void {
  if (canType.value) typeFrom.value = typeFrom.value === null ? within(now.value) : null;
}
/** The stretches inside the time window, with the label that fits on them. */
function typesIn(from: number, to: number, fits: (from: number, to: number, text: string) => boolean) {
  return roadTypes.value.flatMap((item) => {
    const first = Math.max(secondOf(item.from_ns), from);
    const last = Math.min(secondOf(item.to_ns), to);
    if (first >= last) return [];
    const source = typeSource(item);
    const names = item.road_type === null ? source !== 'person' ? [] : ['can’t tell', '?']
      : [ROAD_TYPE_LABELS[item.road_type]?.label.toLowerCase() ?? item.road_type, ROAD_TYPE_LABELS[item.road_type]?.short ?? ''];
    return [{key: item.from_ns, from: first, to: last, cls: 'rt-' + (item.road_type ?? 'unknown'), yours: source === 'person',
      net: source === 'net', none: item.road_type === null && source !== 'person', current: typeFrom.value === null && item === typeHere.value,
      text: names.find((name) => name && fits(first, last, name)) ?? ''}];
  });
}
/** The part of the stretch shown that a chip sets when it is cut, inside the time window: none or one. */
function typeCutIn(from: number, to: number) {
  const here = typeHere.value;
  if (!here || typeCut.value === 'all') return [];
  const [first, last] = typeCut.value === 'before' ? [secondOf(here.from_ns), seconds.value] : [seconds.value, secondOf(here.to_ns)];
  const start = Math.max(first, from);
  const stop = Math.min(last, to);
  return start <= stop ? [{from: start, to: Math.max(stop, start + 0.5)}] : [];
}
/** The span "From here" keeps, to the frame shown, inside the time window: none or one. */
function typeFromIn(from: number, to: number) {
  if (typeFrom.value === null) return [];
  const [first, last] = [typeFrom.value, within(now.value)].sort((a, b) => a - b).map(secondOf) as [number, number];
  const start = Math.max(first, from);
  const stop = Math.min(last, to);
  return start <= stop ? [{from: start, to: Math.max(stop, start + 0.5)}] : [];
}

// -- the signs of Sign review ---------------------------------------------------------------------------------------------
// Not for a guest without the drive's signs. Read again when the page comes back into view: a frame confirmed
// meanwhile shows its boxes.
const given = (kind: string): boolean => props.kinds?.includes(kind) ?? false;
const showSigns = !isGuest() || given('signs');
/** A guest given only the signs: the Signs side alone, the speed's commands off. */
const signsOnly = isGuest() && showSigns && !given('speed');
const signs = shallowRef<SignFrames | null>(null);
const signsError = ref('');
async function readSigns(): Promise<void> {
  if (!showSigns) return;
  try {
    const answer = await api<SignFrames>(address() + '/signs');
    if (!answer || !Array.isArray(answer.frames)) throw new Error('The server sent signs this page cannot read.');
    signs.value = answer;
    signsError.value = '';
  } catch (problem) {
    // A server that does not send them yet says only "Not found.": the page goes on without them, nothing to toast.
    signsError.value = problem instanceof ApiError && problem.status === 404 && problem.message === 'Not found.'
      ? 'The signs are not on this server yet.' : messageOf(problem);
  }
}
const onVisible = (): void => { if (document.visibilityState === 'visible') void readSigns(); };
// The signs on every frame of the video (sign-track.ts, nnslr sign-tracks): read once per drive, behind the same gate as
// Sign review's. A drive without them (404) draws boxes only on Sign review's own frames, one a second.
const track = shallowRef<SignTrack | null>(null);
const trackMissing = ref(false);
const trackError = ref('');
const NO_TRACK = 'This drive has no signs on every frame yet.';
async function readTrack(): Promise<void> {
  if (!showSigns) return;
  try {
    const answer = await api<SignTrack>(address() + '/sign-track');
    if (!answer || !Array.isArray(answer.segments) || !Array.isArray(answer.labels)) throw new Error('The server sent a track this page cannot read.');
    track.value = answer;
    trackMissing.value = false;
    trackError.value = '';
  } catch (problem) {
    // None made yet, or a server without them ("Not found."): the 1 Hz boxes, and a line that says so.
    trackMissing.value = problem instanceof ApiError && problem.status === 404 && (problem.message === NO_TRACK || problem.message === 'Not found.');
    trackError.value = trackMissing.value ? '' : messageOf(problem);
  }
}
/** The video's encode index, from the player: which minutes of the track are the video's. */
const videoIndex = shallowRef<VideoIndex | null>(null);
const trackBoxes = computed(() => track.value && videoIndex.value ? trackFrames(track.value, videoIndex.value) : null);
const reviewed = computed(() => reviewedFrames(signs.value?.frames ?? []));
const ownFrames = computed(() => exactFrames(signs.value?.frames ?? []));
/** The frame the video presents, playing or still (the player's `frame`). */
const presented = shallowRef<VideoFrame | null>(null);
/** Its boxes: of that very frame, never of a time near it, since the car moves the sign within a fraction of a second. */
const frameBoxes = computed(() => boxesAt(presented.value, track.value, trackBoxes.value, reviewed.value, ownFrames.value));
/** The boxes on the picture: those of the frame presented; before the player names one, Sign review's frame on the time
 * shown, as before. */
const pictureBoxes = computed((): Drawn | null => {
  if (presented.value) return frameBoxes.value;
  const frame = signExact.value ? signFrame.value : null;
  return frame && frame.width && frame.height ? {boxes: frame.boxes, width: frame.width, height: frame.height,
    confirmed: frame.confirmed, needsPerson: !!frame.needs_person, frame} : null;
});
/** Whether the drive's boxes are on every frame, in a line of the Signs side. */
const trackLine = computed(() => {
  if (track.value) return 'Boxes on every frame of the video (networks ' + trackNetworks.value + ')'
    + (track.value.current === false ? ' · older networks, or minutes changed since.' : '.');
  if (trackMissing.value) return 'This drive is not read frame by frame yet: boxes only on the frames to review, one a second'
    + ' (Models › Find the signs on every frame).';
  return trackError.value ? 'The boxes on every frame did not load: ' + trackError.value : '';
});
/** The boxes drawn come from the track: the side lists them. */
const drawnTrack = computed(() => !!frameBoxes.value && !frameBoxes.value.frame);
/** The networks that made the track, in a few words (the cutter's run). */
const trackNetworks = computed(() => (track.value?.networks['cutter']?.id ?? '').replace(/^cutter:/, '').replace(/\/signnet$/, ''));
/** The segments of the drive's logs, from the player: what places a frame of Sign review on the drive. */
const segments = shallowRef<SyncSegment[]>([]);
// A frame of Sign review at the time of its very frame of the video (frame 20N of its minute), else of its second.
const placed = computed(() => placeFrames(signs.value?.frames ?? [], (segment, second) =>
  player.value?.timeOfFrame(segment, Math.round(second * 20)) ?? frameSecondOf(segments.value, segment, second)));
/** The frame of Sign review shown, within half a second of the time shown. */
/** A sign of another road is grey: not ours, left out of the lane and of the brain. */
const OTHER_ROAD_COLOR = '#8e8e93';
const signFrame = computed(() => nearestFrame(placed.value, seconds.value));
/** It is the very frame on the picture (a frame of the video or less away), not one near it: only then are its boxes drawn,
 * since a box is where the sign is in that frame and the car moves the sign within a fraction of a second. */
const signExact = computed(() => !!signFrame.value && (presented.value ? signGap.value === 0
  : Math.abs(signFrame.value.at - seconds.value) < 0.05));
/** How many frames of the video the frame of Sign review is after the frame presented (null: another minute, or none). */
const signGap = computed(() => signFrame.value && presented.value?.segment === signFrame.value.segment
  ? Math.round(signFrame.value.second * 20) - presented.value.frame : null);
const runs = computed(() => signRuns(placed.value));
/** The confirmed speed signs inside the time window, their text on the first frame of each where it fits. */
function signsIn(from: number, to: number, x: (second: number) => number) {
  let lastX = -Infinity;
  return runs.value.flatMap((run) => {
    if (run.to < from || run.from > to) return [];
    const start = x(run.from);
    const shown = start - lastX >= run.text.length * 6.5 + 8;
    if (shown) lastX = start;
    return [{key: run.from, at: run.from, ticks: run.ticks.filter((at) => at >= from && at <= to), text: shown ? run.text : '', title: run.text}];
  });
}
/** A tap on a sign of the lane: its first frame, the boxes on the picture. */
const goSign = (at: number): void => player.value?.seek(at);
/** The confirmed speed sign before or after the time shown (its first frame), else null. */
const signBefore = computed(() => [...runs.value].reverse().find((run) => run.from < seconds.value - 0.05) ?? null);
const signAfter = computed(() => runs.value.find((run) => run.from > seconds.value + 0.05) ?? null);
const signWords = (box: {family: string; score?: number | null}): string => signFrame.value?.confirmed || signFrame.value?.needs_person
  ? signFrame.value.reviewer ?? 'confirmed' : typeof box.score === 'number' ? 'model ' + Math.round(box.score * 100) + '%' : 'model';
/** Confirmed, saved by the site's AI reviewer and waiting for a person (as Sign review counts it), or the model's. */
const signState = computed(() => signFrame.value?.confirmed ? 'Confirmed' : signFrame.value?.needs_person ? 'AI · to confirm' : 'Proposed');
/** How far the frame shown is from the time shown, in words. */
const signOffset = computed(() => {
  const gap = signGap.value !== null ? signGap.value / 20 : signFrame.value ? signFrame.value.at - seconds.value : 0;
  return Math.abs(gap) < 0.05 ? 'this frame' : Math.abs(gap).toFixed(1) + ' s ' + (gap > 0 ? 'later' : 'earlier');
});

// The side: the speed (the card as it was) or the signs. Kept for the next drive in this browser.
type Panel = 'speed' | 'signs';
const PANELS: {key: Panel; label: string}[] = [{key: 'speed', label: 'Speed'}, {key: 'signs', label: 'Signs'}];
const PANEL_KEY = 'speed-review-panel';
function readPanel(): Panel {
  try {
    return localStorage.getItem(PANEL_KEY) === 'signs' ? 'signs' : 'speed';
  } catch {
    return 'speed';
  }
}
// A link of the signs opens the Signs side; a guest given only the signs has nothing else.
const panel = ref<Panel>(!showSigns ? 'speed' : signsOnly || asked.side ? 'signs' : readPanel());
watch(panel, (value) => {
  try {
    localStorage.setItem(PANEL_KEY, value);
  } catch {
    // A private window keeps nothing: the speed next time.
  }
});

// -- Sign review's editor, on the Signs side --------------------------------------------------------------------------------
// The frames of Sign review of the drive are edited here as on its page (review/embed.ts, SignEditorPanel.vue,
// SignEditorStage.vue): the frame edited is the very frame the player stands on (its key of Sign review, never the
// nearest second), and a frame opened by the editor (← →, ⇧ ← →, Confirm going on) moves the player there. While the
// player plays, or where Sign review has no frame, the side says so and the boxes are the read-only ones above. A server
// without the editor's data (404), or a failure, leaves the read-only side as it was.
type EditorPhase = 'off' | 'loading' | 'ready' | 'none' | 'failed';
const editor = ref<EditorPhase>('off');
const editorError = ref('');
async function openEditor(): Promise<void> {
  // Also on a read-only panel (actions off): that stops the jobs and the speed's saves, never a confirmation of a frame,
  // which the server takes as on the page of Sign review (a drive Propose again works on is read only there, busy.ts).
  if (!showSigns || editor.value !== 'off' && editor.value !== 'failed') return;
  editor.value = 'loading';
  try {
    const result = await loadSignEditor(props.route);
    if (result !== 'stale') editor.value = result;
  } catch (problem) {
    editor.value = 'failed';
    editorError.value = 'The editor of the signs did not open: ' + messageOf(problem);
  }
}
watch(panel, (value) => { if (value === 'signs') void openEditor(); }, {immediate: true});
onBeforeUnmount(stopSignEditor);
const editing = computed(() => showSigns && panel.value === 'signs' && editor.value === 'ready');
/** The key of Sign review of the frame the player stands on; null while it plays or where Sign review has none. */
const playerKey = computed(() => {
  if (!editing.value) return null;
  const at = player.value?.shownFrame;
  return at ? frameKeyAt(at.segment, at.frame) : null;
});
/** The editor is on screen: the player stands on the frame it has open. */
const onFrame = computed(() => playerKey.value !== null && playerKey.value === editState.currentKey && editRow.value !== null);
/** Where the player has the frame the editor has open (seconds of the drive), null where it has not. */
const timeOfEdited = computed(() => {
  const row = editRow.value;
  const frame = row === null ? null : videoFrameOf(row);
  return row === null || frame === null ? null : player.value?.timeOfFrame(row.segment_index, frame) ?? null;
});
/** The editor on the frame's own picture, the player not standing on it: the drive has no video to show (none on the
 * server, or it could not be made), or the video has not that frame (a minute missing from it). So no frame is out of
 * reach on this side, as on the page of Sign review; ← → go on from frame to frame. */
const alone = computed(() => editing.value && editRow.value !== null && !onFrame.value
  && (!!player.value?.videoGone || videoIndex.value !== null && timeOfEdited.value === null));
watch(playerKey, (key) => {
  if (key !== null && key !== editState.currentKey) openFrame(key, false);
});
watch(() => editState.currentKey, (key) => {
  if (!editing.value || key === null || key === playerKey.value) return;
  const row = rowOf(key);
  const frame = row === null ? null : videoFrameOf(row);
  const at = row === null || frame === null ? null : player.value?.timeOfFrame(row.segment_index, frame) ?? null;
  if (at === null) return;
  player.value?.pause();
  player.value?.seek(at);
});
// The frame a link asked for (?side=signs, &frame=<key>, &segment=N&second=S): its key, else the frame nearest the
// moment, else the first still to review. Opened once the editor has the drive and the player the video's index (the
// player seeks to the moment 2 s early by itself: the frame asked wins).
let wanted = asked.side || asked.frame !== null;
function wantedKey(): string | null {
  if (asked.frame !== null && rowOf(asked.frame) !== null) return asked.frame;
  return linkedFrame() ?? editReport.rows.find((row) => frameStatus(row.frame_key) !== 'done')?.frame_key
    ?? editReport.rows[0]?.frame_key ?? null;
}
watch([editing, videoIndex, segments, () => player.value?.videoGone], () => {
  if (!wanted || !editing.value) return;
  const key = wantedKey();
  const row = key === null ? null : rowOf(key);
  const frame = row === null ? null : videoFrameOf(row);
  const at = row === null || frame === null ? null : player.value?.timeOfFrame(row.segment_index, frame) ?? null;
  // The video's index not in yet: wait for it, unless there is no video to wait for (a frame the video has not is
  // opened in the editor alone, on its own picture).
  if (key !== null && at === null && videoIndex.value === null && !player.value?.videoGone) return;
  wanted = false;
  if (key === null) return;
  openFrame(key, false);
  if (at === null) return;
  player.value?.pause();
  player.value?.seek(at);
}, {immediate: true});
/** "Review with AI" of the drive, on its Signs side: the administrator's, where it can save. */
const canAi = !props.readOnly && !isGuest();
/** The frames left to review, for the sheet. */
const leftToReview = computed(() => editor.value === 'ready'
  ? editReport.rows.filter((row) => frameStatus(row.frame_key) !== 'done').length
  : (signs.value?.frames ?? []).filter((frame) => !frame.confirmed).length);
// A confirmation saved: the lane of the signs and the read-only boxes read again.
let signsTimer: number | undefined;
watch(() => editState.saveCount, () => {
  window.clearTimeout(signsTimer);
  signsTimer = window.setTimeout(() => void readSigns(), 500);
});
onBeforeUnmount(() => window.clearTimeout(signsTimer));

function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement | null;
  if (target?.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey || event.altKey) return;
  // The Signs side: the keys are Sign review's (use-keyboard.ts, while its editor is open), never the speed's, which
  // work only on the Speed side (B there is a block; on the Signs side it is a box).
  if (editing.value || signsOnly || panel.value === 'signs') return;
  if (event.key === 'b') tapBlock();
  else if (event.key === 'Escape' && pending.value !== null) cancelBlock();
  else if (event.key === 'Escape' && typeFrom.value !== null) typeFrom.value = null;
  else if (event.key === 't') startType();
  else if (/^[0-9]$/.test(event.key) && canType.value && !locked.value) {
    const type = event.key === '0' ? null : typeChoices.value[Number(event.key) - 1];
    if (type !== undefined) void setType(type);
  }
  else if (event.key === 'm') mapBlock();
  else if (event.key === 'x') removeHere();
  else if (event.key === 'l') leaveOut();
  else if (event.key === 'w') mark();
  else if (event.key === 'c' && !locked.value && canCheck.value) tapMark('checked');
  else if (event.key === 'k') nextKind();
  else if (event.key === '-') step(-5);
  else if (event.key === '+' || event.key === '=') step(5);
  else if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && (pending.value !== null || typeof block.value?.kph === 'number')) {
    // The arrows step a kph only when there is one to step: else they scroll the page.
    event.preventDefault();
    step(event.key === 'ArrowDown' ? -1 : 1);
  }
}
onMounted(() => {
  window.addEventListener('keydown', onKey);
  document.addEventListener('visibilitychange', onVisible);
  void read();
  void readSigns();
  void readTrack();
});
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey);
  document.removeEventListener('visibilitychange', onVisible);
});

// -- a block's ends dragged on the lane -----------------------------------------------------------------------------------
/** How near on screen an end is pulled onto an edge, and how far the pointer goes before a press is a drag (a tap moves
 * nothing). */
const MAGNET_PX = 8;
const DEAD_PX = 3;
/** A block's end being dragged: the lane's scale when it began, where the pointer and the end were then, the edges that
 * pull it, and where it is now. Where two blocks touch, their shared edge is dragged as one (a roll): item is the block
 * before it, side its end, and roll the block after it. */
interface Drag {
  item: Block; roll: Block | null; side: 'start' | 'end'; pointer: number; down: number; origin: number;
  from: number; to: number; width: number; edges: number[]; ns: number; snapped: number | null; moved: boolean;
}
const dragging = shallowRef<Drag | null>(null);
const canDrag = computed(() => !locked.value && !!drive.value?.span_ns);
/** The block whose end is at a time, the one before an edge two blocks share. */
const endingAt = (ns: number): Block | null => blocks.value.find((other) => other.end_ns === ns) ?? null;
function dragDown(event: PointerEvent, key: number, side: 'start' | 'end', from: number, to: number, width: number): void {
  const grabbed = blocks.value.find((other) => other.start_ns === key);
  const handle = event.currentTarget as SVGElement;
  if (!grabbed || event.button !== 0 || !canDrag.value || width <= 0) return;
  event.preventDefault();
  handle.setPointerCapture(event.pointerId);
  player.value?.pause();
  const before = side === 'start' ? endingAt(grabbed.start_ns) : null;
  const item = before ?? grabbed;
  const roll = before ? grabbed : side === 'end' ? blocks.value.find((other) => other.start_ns === grabbed.end_ns) ?? null : null;
  const edge = before ? 'end' : side;
  const origin = edge === 'start' ? item.start_ns : item.end_ns;
  // Only the edges in the lanes shown pull: the end stays where it can be seen.
  const inside = (ns: number): boolean => ns >= t0.value + from * 1e9 && ns <= t0.value + to * 1e9;
  const edges = [...(player.value?.mapEdges() ?? []).map((second) => t0.value + second * 1e9), now.value].filter(inside);
  dragging.value = {item, roll, side: edge, pointer: event.pointerId, down: event.clientX, origin, from, to, width, edges,
    ns: origin, snapped: null, moved: false};
}
function dragMove(event: PointerEvent): void {
  const drag = dragging.value;
  const span = drive.value?.span_ns;
  if (!drag || drag.pointer !== event.pointerId || !span) return;
  if (!drag.moved && Math.abs(event.clientX - drag.down) < DEAD_PX) return;
  // The end goes as far as the pointer, on the scale of the lanes when it began, and stays inside them: the time shown
  // follows it, and the lanes do not move under it.
  const scale = (drag.to - drag.from) / drag.width * 1e9;
  const room: [number, number] = [Math.max(span[0], t0.value + drag.from * 1e9), Math.min(span[1], t0.value + drag.to * 1e9)];
  const next = draggedEdge(drag.item, drag.side, drag.origin + (event.clientX - drag.down) * scale, blocks.value, room, drag.edges,
    MAGNET_PX * scale, drag.roll);
  if (next.ns === drag.ns && drag.moved) return;
  dragging.value = {...drag, ns: next.ns, snapped: next.snapped, moved: true};
  // The picture of the end: the first frame inside the block at its start, the last one at its end.
  const second = secondOf(next.ns) + (drag.side === 'start' ? 0.001 : -0.001);
  player.value?.seek(Math.min(drag.to, Math.max(drag.from, second)));
}
async function dragUp(event: PointerEvent): Promise<void> {
  const drag = dragging.value;
  if (!drag || drag.pointer !== event.pointerId) return;
  const {item, roll, ns} = drag;
  if (event.type === 'pointerup' && drag.moved && ns !== drag.origin) {
    // Drawn where it was let go until the server answers: the block does not jump back meanwhile. A roll is one write
    // too: the block that grows is moved, and the server cuts the other one where the grown one now covers it.
    if (drag.side === 'start') await send('/blocks', {start_ns: ns, end_ns: item.end_ns, kph: item.kph, note: item.note, from_start_ns: item.start_ns});
    else if (!roll || ns > item.end_ns) await send('/blocks', {start_ns: item.start_ns, end_ns: ns, kph: item.kph, note: item.note, from_start_ns: item.start_ns});
    else await send('/blocks', {start_ns: ns, end_ns: roll.end_ns, kph: roll.kph, note: roll.note, from_start_ns: roll.start_ns});
  }
  if (dragging.value?.pointer === drag.pointer) dragging.value = null;
}
/** A block's ends as drawn: those being dragged where they are now. */
function boundsOf(item: Block): [number, number] {
  const drag = dragging.value;
  if (drag?.roll?.start_ns === item.start_ns) return [drag.ns, item.end_ns];
  if (drag?.item.start_ns !== item.start_ns) return [item.start_ns, item.end_ns];
  return drag.side === 'start' ? [drag.ns, item.end_ns] : [item.start_ns, drag.ns];
}
/** The edge a dragged end is pulled onto, inside the time window: none or one. */
function snapIn(from: number, to: number): number[] {
  const snapped = dragging.value?.snapped;
  if (snapped === null || snapped === undefined) return [];
  const at = secondOf(snapped);
  return at >= from && at <= to ? [at] : [];
}

// -- the lane --------------------------------------------------------------------------------------------------------------
/** The scale of the lane: tens of km/h up to the fastest speed or block of the drive, the same at every zoom. */
const top = computed(() => Math.ceil(Math.max(50, ...speed.value.map((point) => point[1] ?? 0),
  ...blocks.value.map((item) => item.kph ?? 0)) / 10) * 10);
const y = (kph: number): number => TOP + (1 - kph / top.value) * (PLOT - TOP);
const ticks = computed(() => [0, 50, 100, 150].filter((kph) => kph <= top.value));
const end = computed(() => drive.value?.span_ns ? secondOf(drive.value.span_ns[1]) : 0);
/** His speed in the time window as a step line, broken where the logs have none. The lane is drawn again at every
 * frame of the video, the line only when the window moves. */
let drawn: {key: string; path: string} | null = null;
function speedPath(from: number, to: number, width: number, x: (second: number) => number): string {
  const key = [from, to, width, top.value, t0.value].join('|');
  if (drawn?.key === key) return drawn.path;
  const points = speed.value;
  let out = '';
  let open = false;
  for (let index = Math.max(0, speedIndex(points, t0.value + from * 1e9)); index < points.length; index++) {
    const [ns, kph] = points[index]!;
    const at = secondOf(ns);
    if (at > to) break;
    const px = x(Math.max(at, from)).toFixed(1);
    if (kph === null) {
      if (open) out += 'H' + px;
      open = false;
    } else if (!open) {
      out += 'M' + px + ' ' + y(kph).toFixed(1);
      open = true;
    } else {
      out += 'H' + px + 'V' + y(kph).toFixed(1);
    }
  }
  if (open) out += 'H' + x(Math.min(to, end.value)).toFixed(1);
  drawn = {key, path: out};
  return out;
}
watch(drive, () => { drawn = null; });
/** The blocks inside the time window, with what the lane writes on them. */
function heldIn(from: number, to: number) {
  return held.value.flatMap(([start, stop]) => {
    const first = Math.max(secondOf(start), from);
    const last = Math.min(secondOf(stop), to);
    return first < last ? [{key: start, from: first, to: last}] : [];
  });
}
function blocksIn(from: number, to: number) {
  return blocks.value.flatMap((item) => {
    const [start, stop] = boundsOf(item).map(secondOf) as [number, number];
    const first = Math.max(start, from);
    const last = Math.min(stop, to);
    return first < last ? [{key: item.start_ns, from: first, to: last, kph: item.kph, text: item.kph === null ? 'left out' : String(item.kph),
      current: item === block.value, dragged: [dragging.value?.item.start_ns, dragging.value?.roll?.start_ns].includes(item.start_ns),
      // An edge two blocks share has one handle, the end of the block before it: it drags both.
      ends: [...start >= from && !endingAt(item.start_ns) ? [{side: 'start' as const, at: start}] : [],
        ...stop <= to ? [{side: 'end' as const, at: stop}] : []]}] : [];
  });
}
/** The new block still on the page, from its start to the time shown, inside the time window: none or one. */
function pendingIn(from: number, to: number) {
  if (!ends.value || pending.value === null) return [];
  const start = secondOf(pending.value);
  return [{start: start >= from && start <= to ? start : null, from: Math.max(secondOf(ends.value[0]), from),
    to: Math.min(secondOf(ends.value[1]), to)}];
}
/** The part of the time window a person went through for what is picked. */
function checkedIn(from: number, to: number) {
  if (checked.value === null) return [];
  const last = Math.min(secondOf(checked.value), to);
  return from < last ? [{from, to: last}] : [];
}
/** The marks picked inside the time window. */
function worksIn(from: number, to: number) {
  return marks.value.flatMap((item) => {
    const first = Math.max(secondOf(item.start_ns), from);
    const last = Math.min(item.end_ns === null ? end.value : secondOf(item.end_ns), to);
    return first < last ? [{key: item.start_ns, from: first, to: last, open: item.end_ns === null}] : [];
  });
}
</script>

<template>
  <DriveReviewDrive ref="player" class="sr-view" :class="{'sr-signs-on': showSigns && panel === 'signs'}" :route="route" :read-only="readOnly" tab="speedreview" :back-label="isGuest() ? 'My reviews' : 'Speed review'" simple-lanes
                    :arrows="!(editing && onFrame)"
                    sharp :rates="RATES" start-zoom="ten" :car-kph="driverKph" :pick-map="!readOnly" @time="follow"
                    @pick-map="fromMap" @segments="(value) => { segments = value; }" @index="(value) => { videoIndex = value ?? null; }"
                    @frame="(value) => { presented = value; }" @scan="onScan">
    <template #badge><b v-if="drive?.dashcam" class="dashcam-badge">DASHCAM MODE</b><b v-if="drive?.training_ok?.ok"
      class="sr-ok-badge" id="sr-ok-badge">OK FOR TRAINING</b></template>
    <template #hud>
      <span class="dr-box sr-box-label" :class="{empty: label === null}" title="ANSWER: your dashboard speed, or your block where there is one; a dash where openpilot or the cruise held the speed"><small>ANSWER</small>
        <b>{{ label ?? '–' }}</b></span>
    </template>
    <template #picture="{shown, zoomed}">
      <SignEditorStage v-if="editing && (onFrame && shown || alone) && !zoomed" :class="{'sr-editor-alone': alone}"/>
      <svg v-else-if="showSigns && shown && !zoomed && pictureBoxes && pictureBoxes.boxes.length"
           class="sr-sign-boxes" :viewBox="`0 0 ${pictureBoxes.width} ${pictureBoxes.height}`"
           preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <g v-for="(box, index) in pictureBoxes.boxes" :key="index" :style="{color: isOtherRoad(box) ? OTHER_ROAD_COLOR : boxColor(box)}">
          <rect :x="box.bbox_xyxy[0]" :y="box.bbox_xyxy[1]" :width="Math.max(1, box.bbox_xyxy[2] - box.bbox_xyxy[0])"
                :height="Math.max(1, box.bbox_xyxy[3] - box.bbox_xyxy[1])" :class="['sr-sign-box', {proposed: !pictureBoxes.confirmed}]"/>
          <text :x="box.bbox_xyxy[0]" :y="Math.max(pictureBoxes.height / 30, box.bbox_xyxy[1] - pictureBoxes.height / 120)"
                :font-size="pictureBoxes.height / 28" class="sr-sign-tag">{{ isOtherRoad(box) ? 'another · ' + shortName(box) : shortName(box) }}</text>
        </g>
      </svg>
    </template>
    <template #lane="{x, w, width, from, to, fits}">
      <div class="dr-lane sr-lane" title="The answer the brain learns: the line is your dashboard speed; a box is a block of yours, its kph instead of your speed, grey left out, its ends dragged to move them; light grey band: openpilot or the cruise held the speed, not learnt; bar: the marks picked, orange the roadworks, blue a kind of road; grey line: gone through for them">
        <span class="dr-lane-name">Answer<small>km/h</small></span>
        <svg :width="width" :height="HEIGHT" :viewBox="`0 0 ${width} ${HEIGHT}`" aria-hidden="true">
          <g v-for="kph in ticks" :key="kph">
            <line :x1="0" :x2="width" :y1="y(kph)" :y2="y(kph)" class="rg-grid"/>
            <text x="3" :y="y(kph) - 2" class="rg-tick">{{ kph }}</text>
          </g>
          <g v-for="run in heldIn(from, to)" :key="'held' + run.key">
            <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" :height="PLOT - 1" class="sr-held"/>
            <text v-if="fits(run.from, run.to, 'cruise')" :x="x(run.from) + w(run.from, run.to) / 2" :y="PLOT - 4" class="dr-run-text sr-held-text">cruise</text>
          </g>
          <g v-for="run in blocksIn(from, to)" :key="run.key">
            <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" :height="PLOT - 1" rx="3"
                  :class="['sr-block', {'sr-block-out': run.kph === null, 'sr-block-current': run.current || run.dragged}]"/>
            <line v-if="run.kph !== null" :x1="x(run.from)" :x2="x(run.to)" :y1="y(run.kph)" :y2="y(run.kph)" class="sr-block-kph"/>
            <text v-if="fits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="12" class="dr-run-text sr-block-text">{{ run.text }}</text>
          </g>
          <path :d="speedPath(from, to, width, x)" class="rg-line sr-speed"/>
          <g v-for="part in pendingIn(from, to)" :key="'new'">
            <rect v-if="part.from < part.to" :x="x(part.from)" y="1" :width="w(part.from, part.to)" :height="PLOT - 1" rx="3" class="sr-block-new"/>
            <line v-if="part.start !== null" :x1="x(part.start)" :x2="x(part.start)" y1="0" :y2="PLOT" class="sr-block-start"/>
          </g>
          <line v-for="at in snapIn(from, to)" :key="'snap' + at" :x1="x(at)" :x2="x(at)" y1="0" :y2="HEIGHT" class="sr-snap"/>
          <template v-if="canDrag">
            <template v-for="run in blocksIn(from, to)" :key="'ends' + run.key">
              <g v-for="part in run.ends" :key="part.side" class="sr-end" :class="{dragging: run.dragged}"
                 @pointerdown.stop="dragDown($event, run.key, part.side, from, to, width)" @pointermove.stop="dragMove"
                 @pointerup.stop="dragUp" @pointercancel.stop="dragUp">
                <rect :x="x(part.at) - 7" y="1" width="14" :height="PLOT - 1" class="sr-end-hit"/>
                <line :x1="x(part.at)" :x2="x(part.at)" y1="3" :y2="PLOT - 2" class="sr-end-line"/>
              </g>
            </template>
          </template>
          <rect v-for="mark in worksIn(from, to)" :key="mark.key" :x="x(mark.from)" :y="PLOT + 2" :width="w(mark.from, mark.to)" height="4"
                :class="['sr-works', {'sr-works-open': mark.open, 'sr-road': !isWorks}]"/>
          <rect v-for="part in checkedIn(from, to)" :key="part.from" :x="x(part.from)" :y="HEIGHT - 2" :width="w(part.from, part.to)" height="2"
                class="sr-works-checked"/>
        </svg>
      </div>
      <div v-if="roadTypes.length" class="dr-lane sr-type-lane" title="The road type as the brain learns it, a second at a time: light where it is the map's, mid where it is the road labeler's (net), solid where you corrected it, the frame's stretch outlined; you correct it only where it is wrong">
        <span class="dr-lane-name">Road type<small>{{ typeSourceName(typeHere) }}</small></span>
        <svg :width="width" :height="TYPE_HEIGHT" :viewBox="`0 0 ${width} ${TYPE_HEIGHT}`" aria-hidden="true">
          <g v-for="run in typesIn(from, to, fits)" :key="run.key">
            <rect :x="x(run.from)" y="2" :width="w(run.from, run.to)" :height="TYPE_HEIGHT - 4" rx="2"
                  :class="['sr-type', run.cls, {'sr-type-yours': run.yours, 'sr-type-net': run.net, 'sr-type-none': run.none, 'sr-type-current': run.current}]"/>
            <text v-if="run.text" :x="x(run.from) + w(run.from, run.to) / 2" :y="TYPE_HEIGHT / 2 + 3.5"
                  :class="['dr-run-text', 'sr-type-text', {'sr-type-text-yours': run.yours}]">{{ run.text }}</text>
          </g>
          <rect v-for="part in typeFromIn(from, to)" :key="'from'" :x="x(part.from)" y="1" :width="w(part.from, part.to)"
                :height="TYPE_HEIGHT - 2" rx="2" class="sr-type-new"/>
          <rect v-for="part in typeCutIn(from, to)" :key="'cut'" :x="x(part.from)" y="1" :width="w(part.from, part.to)"
                :height="TYPE_HEIGHT - 2" rx="2" class="sr-type-new"/>
        </svg>
      </div>
      <div v-if="signs" class="dr-lane sr-sign-lane" title="The speed signs confirmed (their number, Z a zone, end an end of limit): a tap goes to the sign, its boxes on the picture">
        <span class="dr-lane-name">Signs<small>review</small></span>
        <svg :width="width" :height="SIGN_HEIGHT" :viewBox="`0 0 ${width} ${SIGN_HEIGHT}`" aria-hidden="true">
          <g v-for="sign in signsIn(from, to, x)" :key="sign.key" class="sr-sign-tick" @pointerdown.stop="goSign(sign.at)">
            <rect :x="x(sign.at) - 6" y="0" :width="Math.max(12, w(sign.at, sign.ticks[sign.ticks.length - 1] ?? sign.at) + 12)"
                  :height="SIGN_HEIGHT" class="sr-sign-hit"/>
            <line v-for="at in sign.ticks" :key="at" :x1="x(at)" :x2="x(at)" :y1="SIGN_HEIGHT - 9" :y2="SIGN_HEIGHT - 1"/>
            <text v-if="sign.text" :x="x(sign.at)" :y="SIGN_HEIGHT - 11">{{ sign.text }}</text>
            <title>{{ sign.title }}</title>
          </g>
        </svg>
      </div>
    </template>
    <template #side>
      <section class="sr-card" aria-labelledby="sr-card-title">
        <Segmented v-if="showSigns && !signsOnly" v-model="panel" :options="PANELS" label="The side: the speed or the signs" class="sr-panel"/>
        <SpeedTrainingSwitch v-if="drive && !signsOnly" :drive="drive" :read-only="readOnly" @changed="show" @failed="read"/>
        <template v-if="panel === 'signs' && editing">
          <h3 id="sr-card-title">The signs at {{ clockOf(seconds) }}</h3>
          <p v-if="alone" class="dr-hint" id="sr-editor-alone">{{ player?.videoGone ? 'The video of the drive cannot be shown'
            : 'The video of the drive has not this frame' }}: its own picture is on screen. ← → go on from frame to frame.</p>
          <SignEditorPanel v-if="onFrame || alone"/>
          <p v-else-if="player?.playing" class="dr-hint" id="sr-editor-playing">Playing. Pause on a frame (Space) to edit its
            signs; ← → then go frame by frame.</p>
          <p v-else class="dr-hint" id="sr-editor-none">No frame to review at this moment of the drive.</p>
          <p v-if="trackLine" class="dr-hint sr-track-line">{{ trackLine }}</p>
        </template>
        <template v-else-if="panel === 'signs'">
          <h3 id="sr-card-title">The signs at {{ clockOf(seconds) }}</h3>
          <p v-if="editor === 'failed'" class="dr-hint">{{ editorError }}</p>
          <p v-if="signsError" class="dr-hint" id="sr-signs-missing">{{ signsError }}</p>
          <p v-else-if="!signs" class="dr-hint">Loading…</p>
          <template v-else>
            <p v-if="trackLine" class="dr-hint sr-track-line" id="sr-track-line">{{ trackLine }}</p>
            <template v-if="drawnTrack && frameBoxes && presented">
              <p class="sr-sign-frame">Frame {{ presented.segment }}/{{ presented.frame }} of the video<span class="sr-tag sr-tag-out">Networks</span></p>
              <ul v-if="frameBoxes.boxes.length" class="sr-sign-list" id="sr-sign-list">
                <li v-for="(box, index) in frameBoxes.boxes" :key="index" :class="{speed: isSpeedSign(box)}">
                  <i class="sr-sign-dot proposed" :style="{color: boxColor(box)}"></i>
                  <span>{{ describe(box) }}</span><small>{{ typeof box.score === 'number' ? 'model ' + Math.round(box.score * 100) + '%' : 'model' }}</small>
                </li>
              </ul>
              <p v-else class="dr-hint">The networks found no sign on this frame.</p>
            </template>
            <template v-else-if="signFrame">
              <p class="sr-sign-frame">Frame at {{ clockOf(signFrame.at) }} · {{ signOffset }}<span class="sr-tag"
                :class="{'sr-tag-out': !signFrame.confirmed}">{{ signState }}</span></p>
              <ul v-if="signFrame.boxes.length" class="sr-sign-list" id="sr-sign-list">
                <li v-for="(box, index) in signFrame.boxes" :key="index" :class="{speed: isSpeedSign(box)}">
                  <i class="sr-sign-dot" :class="{proposed: !signFrame.confirmed}" :style="{color: isOtherRoad(box) ? OTHER_ROAD_COLOR : boxColor(box)}"></i>
                  <span>{{ describe(box) }}{{ isOtherRoad(box) ? ' · another road' : '' }}</span><small>{{ signWords(box) }}</small>
                </li>
              </ul>
              <p v-else class="dr-hint">{{ signFrame.confirmed ? 'Confirmed with no sign on it.' : signFrame.needs_person ? 'The AI reviewer found no sign here: a person has not confirmed it yet.' : 'The model proposed no sign here.' }}</p>
            </template>
            <p v-else class="dr-hint">No frame to review within half a second of {{ clockOf(seconds) }}.</p>
            <div class="sr-tools">
              <button type="button" class="btn small" id="sr-sign-before" :disabled="!signBefore" @click="signBefore && goSign(signBefore.from)">‹ Sign before</button>
              <button type="button" class="btn small" id="sr-sign-after" :disabled="!signAfter" @click="signAfter && goSign(signAfter.from)">Sign after ›</button>
            </div>
            <p class="dr-hint">Solid: confirmed; dashed: the model's proposals, or the AI reviewer's boxes, on a frame not confirmed
              yet.<template v-if="editor === 'loading'"> The editor is
              opening…</template></p>
          </template>
        </template>
        <p v-else-if="error" class="dr-hint">{{ error }}</p>
        <p v-else-if="!drive" class="dr-hint">Loading…</p>
        <template v-else-if="!drive.span_ns">
          <p class="dr-hint" id="sr-no-speed">{{ noSpeed }}</p>
        </template>
        <template v-else>
          <h3 id="sr-card-title">The answer at {{ clockOf(seconds) }}</h3>
          <p v-if="drive.dashcam === false" class="dr-hint" id="sr-not-learnt"><strong>sunnypilot was on in this drive: the brain
            does not learn from it.</strong> It learns only drives with sunnylink › Toggles › Enable sunnypilot off.</p>
          <p class="dr-hint">ANSWER is your dashboard speed (vEgoCluster), or your block where there is one. Where openpilot or
            the cruise held the speed it is a dash: not your foot, not learnt.</p>
          <div class="sr-value">
            <span>{{ block ? 'Block' : 'Your dashboard speed' }}</span><b>{{ (machine ? driverKph : label) ?? '–' }}</b>
            <span v-if="(machine ? driverKph : label) !== null">km/h</span>
            <span v-if="block" class="sr-tag" :class="{'sr-tag-out': block.kph === null}">{{ block.kph === null ? 'Left out' : 'Your block' }}</span>
            <span v-else-if="machine" class="sr-tag sr-tag-out">Not learnt</span>
          </div>
          <p v-if="block" class="dr-hint">{{ clockOf(secondOf(block.start_ns)) }}–{{ clockOf(secondOf(block.end_ns)) }} ·
            your dashboard speed {{ driverKph ?? '–' }} km/h<template v-if="block.reviewer"> · {{ block.reviewer }}</template></p>
          <p v-else-if="machine" class="dr-hint">openpilot or the car's cruise held the speed here, not your foot:
            the training leaves it out, unless a block of yours says a speed.</p>

          <div v-if="pending !== null" class="sr-new" role="group" aria-labelledby="sr-new-title">
            <p id="sr-new-title" class="sr-new-title">New block from {{ clockOf(secondOf(pending)) }}<template v-if="canEnd && ends">
              to {{ clockOf(secondOf(pending === ends[0] ? ends[1] : ends[0])) }}</template></p>
            <div class="sr-kph" role="group" aria-label="The speed of the new block">
              <button type="button" class="btn small" :disabled="pendingOut" title="Key −" @click="step(-5)">−5</button>
              <button type="button" class="btn small" :disabled="pendingOut" title="Key ↓" @click="step(-1)">−1</button>
              <input class="sr-kph-value" type="number" inputmode="numeric" min="1" :max="maxKph"
                     :value="draftOf('new') ?? (pendingOut ? '' : pendingKph)" :placeholder="pendingOut ? 'left out' : ''" :disabled="pendingOut"
                     aria-label="The speed of the new block, km/h" enterkeyhint="done" @focus="player?.pause()" @input="typing('new', $event)"
                     @change="typed" @blur="untyped" @keydown.enter="($event.target as HTMLInputElement).blur()">
              <button type="button" class="btn small" :disabled="pendingOut" title="Key ↑" @click="step(1)">+1</button>
              <button type="button" class="btn small" :disabled="pendingOut" title="Key +" @click="step(5)">+5</button>
            </div>
            <div class="sr-tools">
              <button type="button" class="btn small" id="sr-block-out" :aria-pressed="pendingOut"
                      title="These moments stay out of the training (key l)" @click="pendingOut = !pendingOut">Leave out</button>
              <button type="button" class="btn small" id="sr-block-cancel" title="Esc" @click="cancelBlock">Cancel</button>
            </div>
            <button type="button" class="btn primary sr-wide" id="sr-block-end" :disabled="locked || !canEnd" title="Key b" @click="endBlock">Block to here</button>
            <p v-if="!canEnd" class="dr-hint">Go to the end of the block: it lasts at least 1 s.</p>
          </div>
          <template v-else>
            <button type="button" class="btn primary sr-wide" id="sr-block-start" :disabled="locked || !canStart" title="Key b"
                    @click="startBlock">Block from here</button>
            <p v-if="!block" class="dr-hint">A block gives its own speed instead of yours: from here to where you go, with the speed typed or
              stepped. A tap on a limit of the Map lane makes one with it; drag the ends of a box on the lane to move them.</p>
          </template>

          <template v-if="block && pending === null">
            <div class="sr-kph" role="group" aria-label="The speed of this block">
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key −" @click="step(-5)">−5</button>
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key ↓" @click="step(-1)">−1</button>
              <input class="sr-kph-value" type="number" inputmode="numeric" min="1" :max="maxKph"
                     :value="draftOf(block.start_ns) ?? block.kph ?? ''" :placeholder="block.kph === null ? 'left out' : ''"
                     :disabled="locked || block.kph === null" aria-label="The speed of this block, km/h" enterkeyhint="done"
                     @focus="player?.pause()" @input="typing(block.start_ns, $event)" @change="typed"
                     @blur="untyped" @keydown.enter="($event.target as HTMLInputElement).blur()">
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key ↑" @click="step(1)">+1</button>
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key +" @click="step(5)">+5</button>
            </div>
            <div class="sr-tools">
              <button v-if="block.kph !== null" type="button" class="btn small" id="sr-block-leave" :disabled="locked" title="Key l"
                      @click="setBlock(block, null)">Leave out</button>
              <button v-else type="button" class="btn small" id="sr-block-give" :disabled="locked" title="Key l"
                      @click="setBlock(block, kphOf(block))">Give a speed ({{ kphOf(block) }})</button>
              <button type="button" class="btn small" id="sr-block-remove" :disabled="locked" title="Key x" @click="removeBlock(block)">Remove block</button>
            </div>
          </template>

          <div v-if="blocks.length" class="sr-blocks">
            <p class="sr-blocks-title">{{ plural(blocks.length, 'block') }}</p>
            <ul>
              <li v-for="item in blocks" :key="item.start_ns" :class="{current: item === block}">
                <button type="button" class="sr-block-go" @click="go(item)"><span>{{ clockOf(secondOf(item.start_ns)) }}–{{ clockOf(secondOf(item.end_ns)) }}</span>
                  <b>{{ blockText(item) }}</b></button>
                <button type="button" class="btn small sr-block-x" :disabled="locked" :aria-label="'Remove the block from ' + clockOf(secondOf(item.start_ns))"
                        @click="removeBlock(item)">✕</button>
              </li>
            </ul>
          </div>

          <div class="sr-road-card" role="group" aria-labelledby="sr-road-title">
            <p id="sr-road-title" class="sr-blocks-title">The road</p>
            <div v-if="roadTypes.length" class="sr-type-row">
              <p class="sr-type-now">Road type <b>{{ typeHere ? typeName(typeHere) : '–' }}</b><span v-if="typeHere"
                class="sr-tag" :class="{'sr-tag-out': typeSource(typeHere) !== 'person'}"
                :title="typeSource(typeHere) === 'net' ? 'The road labeler’s' + (typeHere.run ? ', run ' + typeHere.run : '') : undefined">{{ TYPE_SOURCE_TAGS[typeSource(typeHere)] }}</span></p>
              <template v-if="typeCuttable">
                <p class="sr-cut-head"><strong>✂ Cut here?</strong> The stretch goes on before and after this frame: a type sets
                  the whole of it, or only the part before or after the frame.</p>
                <Segmented v-model="typePart" :options="TYPE_PARTS" label="Cut the stretch at this frame" class="sr-type-part"/>
              </template>
              <div v-if="canType" class="sr-kinds sr-types" role="group" aria-label="The road type">
                <button v-for="(type, index) in typeChoices" :key="type ?? 'none'" type="button" class="sr-kind sr-type-chip"
                        :disabled="locked" :aria-current="typeOn(type) ? 'true' : undefined"
                        :title="(typeFrom !== null ? 'Sets it from ' + clockOf(secondOf(typeFrom)) + ' to the frame shown' : typeCut === 'before' ? 'Sets it from the start of the stretch to the frame shown' : typeCut === 'after' ? 'Sets it from the frame shown to the end of the stretch' : 'Sets it on the whole stretch of the frame shown') + ' (key ' + (type === null ? 0 : index + 1) + ')'"
                        @click="setType(type)"><i class="sr-type-dot" :class="'rt-' + (type ?? 'unknown')"/>{{ roadTypeLabel(type) }}</button>
                <button type="button" class="sr-kind sr-type-from" id="sr-type-from" :aria-pressed="typeFrom !== null" :disabled="locked"
                        title="Key t: then a type sets it from here to the frame you go to" @click="startType">{{ typeFrom === null ? 'From here…' : 'Cancel' }}</button>
              </div>
              <p class="dr-hint"><template v-if="typeFrom !== null">From {{ clockOf(secondOf(typeFrom)) }} to the frame shown: tap its type.</template><template
                v-else-if="typeHere">{{ typeName(typeHere) }} {{ clockOf(secondOf(typeHere.from_ns)) }}–{{ clockOf(secondOf(typeHere.to_ns)) }}<template
                v-if="typeHere.reviewer"> · {{ typeHere.reviewer }}</template><template
                v-else-if="typeSource(typeHere) === 'net'"> · road labeler{{ typeHere.run ? ' ' + typeHere.run : '' }}</template><template v-if="canType"> · a type sets it {{ typeCut === 'before' ? 'from its start to ' + clockOf(seconds)
                : typeCut === 'after' ? 'from ' + clockOf(seconds) + ' to its end' : 'on this stretch' }}</template></template><template
                v-else>The map has no type here.</template></p>
            </div>
            <p v-else-if="drive.road_type_names && !readOnly && !isGuest()" class="dr-hint">No road types for this drive yet: Read the map in Data › Prepare.</p>
            <div class="sr-kinds" role="group" aria-label="Kinds of road: a tap where it starts, a tap where it ends">
              <button v-for="item in kinds" :key="item.kind" type="button" class="sr-kind"
                      :class="{'sr-kind-works': item.kind === ROADWORKS, 'sr-kind-here': inside.has(item.kind)}"
                      :aria-current="item.kind === pickedKind.kind" :aria-pressed="inside.has(item.kind)" :disabled="locked"
                      :title="inside.has(item.kind) ? 'Marked here' : 'A tap: it starts here (key w)'"
                      @click="tapKind(item.kind)">{{ item.label }}</button>
              <button v-if="canAdd && !adding" type="button" class="sr-kind sr-kind-add" id="sr-kind-add" aria-label="Add a kind of road"
                      @click="adding = true">+</button>
            </div>
            <form v-if="adding" class="sr-kind-new" @submit.prevent="addKind">
              <input v-model="newLabel" type="text" maxlength="40" placeholder="Bumpy road, cobbles, …" aria-label="The name of the new kind of road"
                     enterkeyhint="done" autofocus>
              <button type="submit" class="btn small primary" :disabled="saving || !newLabel.trim()">Add</button>
              <button type="button" class="btn small" @click="adding = false; newLabel = ''">Cancel</button>
            </form>
            <p v-if="markOpen" class="sr-open" id="sr-works-open"><strong>{{ pickedKind.label }} from {{ clockOf(secondOf(markOpen.start_ns)) }}</strong>:
              go to where it ends and tap {{ pickedKind.label }} again.</p>
            <div class="sr-tools">
              <button v-if="markHere" type="button" class="btn small" id="sr-works-remove" :disabled="locked" @click="tapMark('remove')">Remove {{ pickedKind.label }} here</button>
              <button type="button" class="btn small" id="sr-works-checked" :disabled="locked || !canCheck"
                      :title="'From the start of the drive to this frame every ' + pickedKind.label.toLowerCase() + ' is marked: no mark there means none (key c)'"
                      @click="tapMark('checked')">Checked to here</button>
            </div>
            <p class="dr-hint">A tap on a kind where it starts, a tap where it ends. {{ pickedKind.label }}<template v-if="markShown"> {{ clockOf(secondOf(markShown.start_ns)) }}{{ markShown.end_ns === null ? ', no end yet' : '–' + clockOf(secondOf(markShown.end_ns)) }}</template><template
              v-else>: no mark here</template><template v-if="checked !== null"> · checked up to {{ clockOf(secondOf(checked)) }}</template><template
              v-else> · not checked yet</template></p>
            <p v-if="!isWorks" class="dr-hint">A kind of road changes no speed: the brain learns it aside, as the roadworks, to read the road better.</p>
          </div>
          <details v-if="!readOnly" class="dr-more sr-keys"><summary>Keys</summary>
            <p class="dr-hint">b block from / to here · m block of the map · x remove the block · l leave out · ↑ ↓ ±1 · − + ±5 ·
              Esc cancel · k next kind of road · w start / end of the last one tapped · c checked to here · t road type from here · 1–9 its type,
              0 can't tell · space play · ← → a frame · ⇧ ← → 1 s</p></details>
        </template>
        <div v-if="canAi && panel === 'signs'" class="sr-ai">
          <button type="button" class="btn small ai-route" id="sr-ai" :data-ai-route="route" :data-ai-name="route"
                  :data-ai-left="leftToReview">Review with AI</button>
          <p class="dr-hint">The AI reviewer of Settings tries or saves the boxes of the frames left; you confirm what it saves.</p>
        </div>
      </section>
      <Teleport v-if="canAi" to="body"><AiRouteSheet/></Teleport>
    </template>
  </DriveReviewDrive>
</template>
