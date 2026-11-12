<script setup lang="ts">
// The comma's screen over the picture of a drive (Models › Drive, DriveSimulation.vue): the two squares of the car's
// screen as the comma draws them (sunnypilot's selfdrive/ui/sunnypilot/mici/onroad: vision_speed_limit.py,
// sign_reading.py), in the comma's own pixels, 476 × 240, and scaled as a whole (scale). On top the vision square: the
// speed to hold in fuchsia under an eye, dark with a dash without one. Under it the reading square: the classifier's
// confidence and the last limit read as written (END), no metres; its border is red while the sign is in view, and it
// blinks, a white pulse growing under it, from the moment the network sees a sign until the car passes it. A road sign read (its short name, crossed out at its end)
// goes in a small box under the square, red while in view, until the next sign, and leaves the limit in it. Left of
// it the kind of road the network sees. The page gives what the car's message had at the time shown and the pulse of
// that time (0 to 1): nothing here moves by itself, so a paused video is a still screen. The page places it (right,
// top); its colours and the comma's own font, CommaInter, are control.html's (.cms-…).
import {computed, ref} from 'vue';
import {EYE_ICON} from './drive-review';
import {isRoadSign, pyRound, readingText, SCREEN} from './simulation';

const props = defineProps<{limitKph: number; rawClass: string; rawValue: number; confidence: number; limitClass: string;
  limitValue: number; limitConfidence: number; seen: boolean; distance: number; road: string; pulse: number; scale: number}>();

// The comma's numbers (vision_speed_limit.py): two squares in the top right corner, the reading under the vision.
const SIZE = 72;
const MARGIN = 12;
const GAP = 8;
const BORDER = 3;
/** ROUNDNESS 0.3 of half the side. */
const RADIUS = 0.15 * SIZE;
const LEFT = SCREEN.width - MARGIN - SIZE;
const MIDDLE = LEFT + SIZE / 2;
const VISION_TOP = MARGIN;
const READING_TOP = MARGIN + SIZE + GAP;
/** The eye: eye_fill.png (300 × 213) loaded into 24 × 17 keeping its shape is 23 × 17 (application.py
 * _load_image_from_path), centred in the square; the path is drawn in a box of 24 × 17. */
const EYE = {width: 23, height: 17, top: 9};
/** raylib's border (DrawRectangleRoundedLinesEx) lies outside the square: its band runs from 2.5 px out to 0.5 px in,
 * so the stroke of 3 px goes on a rect 1 px bigger on every side, its corners 1.5 px rounder. */
const OUT = 1;
const OUT_ROUND = BORDER / 2;
/** The comma scales every font size by this (system/ui/lib/application.py FONT_SCALE): a text of size F is a line of
 * h = F × FONT_SCALE px. raylib sizes Inter by its line, ascent to descent (stbtt_ScaleForPixelHeight: 2728 + 680 of its
 * 2816 units an em), and puts the baseline ASCENT of h under the y it is given; an SVG font-size is the em, EM of h. */
const FONT_SCALE = 1.16;
const EM = 2816 / 3408;
const ASCENT = 2728 / 3408;
/** A line of the reading square, h high, by the middle of its capitals and digits from the square's top: its top at
 * CAP_MIDDLE of h above it (sign_reading._draw_line), its baseline ASCENT of h under that. */
const CAP_MIDDLE = 0.63;
const readingBaseline = (middle: number, h: number): number => READING_TOP + middle + (ASCENT - CAP_MIDDLE) * h;
/** The reading square's two lines, by the middle of their capitals from its top (sign_reading.py). */
const SMALL = 16 * FONT_SCALE;
const CONFIDENCE_MIDDLE = 17;
const TEXT_MIDDLE = 44;
/** The sign read: its size by its length, then shrunk to the room between the borders. */
const TEXT_SIZES: Record<number, number> = {1: 28, 2: 28, 3: 26};
const TEXT_WIDTH = SIZE - 2 * (BORDER + 4);
/** The kind of road, its line level with the middle of the reading square (vision_speed_limit.road_type_place). */
const ROAD = 20 * FONT_SCALE;
const ROAD_BASELINE = READING_TOP + (SIZE - ROAD) / 2 + ASCENT * ROAD;
/** A sign read: the square grows by this share of its side and back. */
const PULSE_GROW = 0.18;
/** A road sign read: a box under the reading square, as wide (sign_reading ROAD_GAP, ROAD_HEIGHT, ROAD_FONT_SIZE). */
const ROAD_GAP = 4;
const ROAD_HEIGHT = 24;
const ROAD_TOP = READING_TOP + SIZE + ROAD_GAP;
const ROAD_SIGN = 18 * FONT_SCALE;

/** The width of a bold text whose line is h px, as raylib's MeasureTextEx gives it, by which the comma shrinks the sign
 * read to the room of its square. raylib loads the font at 200 px (application.py _load_fonts) and keeps each letter's
 * advance there as whole px ((int)(advance × scale), LoadFontData); MeasureTextEx adds them up and scales the sum to the
 * line asked. Each letter's advance in the font's units is measured once on a canvas at a size of one px a unit, in
 * the comma's font (control.html's CommaInter), where the browser rounds nothing; without a canvas, about a bold digit's.
 * Until the font has loaded the canvas measures another one: once it has, every letter is measured again (fontLoaded,
 * read by reading). */
const FAMILY = 'CommaInter';
const UNITS = 2816;
const BASE = 200;
const BASE_SCALE = Math.fround(BASE / 3408);
function newRuler(): CanvasRenderingContext2D | null {
  const context = document.createElement('canvas').getContext('2d');
  if (context) {
    // As .cms in control.html: the comma's own font, else the page's; letter by letter, so never a kerning.
    const fallback = getComputedStyle(document.documentElement).getPropertyValue('--font-ui').trim() || 'sans-serif';
    context.font = '700 ' + UNITS + 'px ' + FAMILY + ',' + fallback;
  }
  return context;
}
let ruler: CanvasRenderingContext2D | null | undefined;
const advances = new Map<string, number>();
const fontLoaded = ref(0);
function advanceOf(letter: string): number {
  const known = advances.get(letter);
  if (known !== undefined) return known;
  if (ruler === undefined) {
    ruler = newRuler();
    // A new canvas once the font is there: one made before it may keep the font it fell back to.
    document.fonts?.load('700 16px ' + FAMILY).then(() => {
      ruler = newRuler();
      advances.clear();
      fontLoaded.value += 1;
    }, () => undefined);
  }
  const units = ruler ? Math.round(ruler.measureText(letter).width) : 0.62 * UNITS;
  // In float, as raylib's C.
  const advance = Math.trunc(Math.fround(Math.fround(units) * BASE_SCALE));
  advances.set(letter, advance);
  return advance;
}
function textWidth(text: string, h: number): number {
  let sum = 0;
  for (const letter of text) sum += advanceOf(letter);
  return sum * h / BASE;
}

/** The speed to hold: its line centred between the eye and the foot of the square (vision_speed_limit._render). */
const vision = computed(() => {
  const text = props.limitKph > 0 ? String(props.limitKph) : '-';
  const h = (text.length < 3 ? 34 : 28) * FONT_SCALE;
  const top = VISION_TOP + EYE.top + EYE.height;
  return {text, size: h * EM, y: top + (VISION_TOP + SIZE - top - h) / 2 + ASCENT * h};
});

/** A line of text centred at the square's middle, its capitals' middle at middle (from the screen's top), shrunk to
 * the room between the borders; with the line through it when crossed, from 2 px before its first letter, a third of
 * its line under the middle of its capitals, to 2 px after its last, a third above (sign_reading._draw_line). */
function line(text: string, full: number, middle: number, crossed: boolean) {
  const wide = textWidth(text, full);
  const h = wide > TEXT_WIDTH ? full * TEXT_WIDTH / wide : full;
  const half = Math.min(wide, TEXT_WIDTH) / 2;
  return {text, size: h * EM, y: middle + (ASCENT - CAP_MIDDLE) * h,
    cross: crossed ? {x1: MIDDLE - half - 2, y1: middle + h / 3, x2: MIDDLE + half + 2, y2: middle - h / 3} : null};
}

const reading = computed(() => {
  void fontLoaded.value;
  const {text, crossed} = readingText(props.limitClass, props.limitValue);
  const shown = text || '-';
  return {
    ...line(shown, (TEXT_SIZES[shown.length] ?? 22) * FONT_SCALE, READING_TOP + TEXT_MIDDLE, crossed), known: text !== '',
    confidence: text ? pyRound(props.limitConfidence * 100) + '%' : '',
  };
});
/** The road sign read, in its box under the square; null after any other reading. */
const roadSign = computed(() => {
  void fontLoaded.value;
  if (!isRoadSign(props.rawClass)) return null;
  const {text, crossed} = readingText(props.rawClass, props.rawValue);
  return line(text, ROAD_SIGN, ROAD_TOP + ROAD_HEIGHT / 2, crossed);
});

/** The pulse: the reading square's outline grown and white, drawn under it; its alphas whole steps of 255, as the
 * comma's colours (sign_reading._render: int(70 × pulse), int(255 × pulse)). */
const halo = computed(() => {
  if (props.pulse <= 0) return null;
  const grow = SIZE * PULSE_GROW * props.pulse;
  return {x: LEFT - grow / 2, y: READING_TOP - grow / 2, side: SIZE + grow, fill: Math.trunc(70 * props.pulse) / 255,
    line: Math.trunc(255 * props.pulse) / 255};
});
</script>

<template>
  <svg class="cms" :width="SCREEN.width * scale" :height="SCREEN.height * scale" :viewBox="`0 0 ${SCREEN.width} ${SCREEN.height}`"
       aria-hidden="true">
    <!-- The reading square first, as on the comma: its pulse slides under the vision square. Each border is drawn on
         the rect raylib's lies on, OUT px outside its square. -->
    <g v-if="halo" class="cms-pulse">
      <rect :x="halo.x" :y="halo.y" :width="halo.side" :height="halo.side" :rx="0.15 * halo.side" :fill-opacity="halo.fill"/>
      <rect class="cms-line" :x="halo.x - OUT" :y="halo.y - OUT" :width="halo.side + 2 * OUT" :height="halo.side + 2 * OUT"
            :rx="0.15 * halo.side + OUT_ROUND" :stroke-opacity="halo.line"/>
    </g>
    <rect class="cms-reading" :x="LEFT" :y="READING_TOP" :width="SIZE" :height="SIZE" :rx="RADIUS"/>
    <rect class="cms-line cms-reading-line" :class="{seen: seen && !roadSign}" :x="LEFT - OUT" :y="READING_TOP - OUT" :width="SIZE + 2 * OUT"
          :height="SIZE + 2 * OUT" :rx="RADIUS + OUT_ROUND"/>
    <g class="cms-text" :class="{none: !reading.known}">
      <text v-if="reading.confidence" :x="MIDDLE" :y="readingBaseline(CONFIDENCE_MIDDLE, SMALL)" :font-size="SMALL * EM">{{ reading.confidence }}</text>
      <text :x="MIDDLE" :y="reading.y" :font-size="reading.size">{{ reading.text }}</text>
      <line v-if="reading.cross" class="cms-cross" v-bind="reading.cross"/>
    </g>
    <g v-if="roadSign">
      <rect class="cms-reading" :x="LEFT" :y="ROAD_TOP" :width="SIZE" :height="ROAD_HEIGHT" :rx="0.15 * ROAD_HEIGHT"/>
      <rect class="cms-line cms-reading-line" :class="{seen}" :x="LEFT - OUT" :y="ROAD_TOP - OUT" :width="SIZE + 2 * OUT"
            :height="ROAD_HEIGHT + 2 * OUT" :rx="0.15 * ROAD_HEIGHT + OUT_ROUND"/>
      <g class="cms-text">
        <text :x="MIDDLE" :y="roadSign.y" :font-size="roadSign.size">{{ roadSign.text }}</text>
        <line v-if="roadSign.cross" class="cms-cross" v-bind="roadSign.cross"/>
      </g>
    </g>

    <rect class="cms-vision" :class="{none: !limitKph}" :x="LEFT" :y="VISION_TOP" :width="SIZE" :height="SIZE" :rx="RADIUS"/>
    <rect class="cms-line cms-vision-line" :class="{none: !limitKph}" :x="LEFT - OUT" :y="VISION_TOP - OUT" :width="SIZE + 2 * OUT"
          :height="SIZE + 2 * OUT" :rx="RADIUS + OUT_ROUND"/>
    <g class="cms-text" :class="{none: !limitKph}">
      <path :d="EYE_ICON" :transform="`translate(${LEFT + (SIZE - EYE.width) / 2} ${VISION_TOP + EYE.top}) scale(${EYE.width / 24} 1)`"
            fill-rule="evenodd"/>
      <text :x="MIDDLE" :y="vision.y" :font-size="vision.size">{{ vision.text }}</text>
    </g>
    <!-- The kind of road, left of the reading square and level with its middle. -->
    <text v-if="road" class="cms-road" :x="LEFT - GAP" :y="ROAD_BASELINE" :font-size="ROAD * EM">{{ road }}</text>
  </svg>
</template>
