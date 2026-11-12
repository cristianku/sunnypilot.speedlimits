<script setup lang="ts">
// The runs, newest first: name and when, where the weights started from, the state, and the headline numbers on
// the test (the reader's accuracy on every crop and on the crops a person confirmed, the detector's recall and
// precision; for a run of the classifier with “not a sign”, its signs read right, the “not a sign” crops it rejects and
// the chain end to end, in one line). A row opens the run below; the name is the button, for the keyboard.
import {thousands} from '../common/format';
import type {RunSummary} from '../common/panel-types';
import {RUN_STATE, modeText, notEvaluatedText, pct, runWhen, stageText, usernetLine} from './evaluate';
import {perFrame} from './models';

/** A cutter run in one line: a SignNet as the car reads it, "Cutter (SignNet), as the car reads: 40.0% of the signs found
 * (speed signs 45.0%) · 0.22 false boxes per frame"; an old Detector at its last size, "Cutter 960 × 544: 81.0% of the
 * signs found at score 0.1 (speed signs 83.0%) · 2.10 boxes per frame". */
function cutterLine(run: RunSummary): string {
  const line = run.cutters?.[run.cutters.length - 1];
  if (!line) return 'Cutter: open the run for its numbers.';
  if (line.kind === 'signnet') {
    if (!line.evaluated) return 'Cutter (SignNet): not measured as the car reads it yet; open the run for its numbers.';
    return 'Cutter (SignNet), as the car reads (best ' + (line.max_boxes ?? 5) + ' at ' + (line.score_threshold ?? 0.3) + '): '
      + pct(line.recall) + ' of the ' + (line.families === 'speed' ? 'speed signs' : 'signs') + ' found'
      + (line.families === 'all' && line.speed_recall != null ? ' (speed signs ' + pct(line.speed_recall) + ')' : '')
      + ' · ' + perFrame(line.false_per_frame) + ' false boxes per frame';
  }
  return 'Cutter ' + (line.size ?? '–').replace('x', ' × ') + ': ' + pct(line.recall) + ' of the signs found at score ' + (line.score_threshold ?? '–')
    + (line.families === 'all' ? ' (speed signs ' + pct(line.speed_recall) + ')' : '') + ' · ' + perFrame(line.boxes_per_frame) + ' boxes per frame';
}

/** A classifier run in one line: "Classifier 91.2% of the signs · rejects 88.0% of “not a sign” · chain 72.0% end to end,
 * 0.31 false per frame". */
function classifierLine(run: RunSummary): string {
  const parts: string[] = [];
  if (run.classifier?.accuracy != null) parts.push('Classifier ' + pct(run.classifier.accuracy) + ' of the signs');
  if (run.classifier?.rejection_rate != null) parts.push('rejects ' + pct(run.classifier.rejection_rate) + ' of “not a sign”');
  if (run.chain) parts.push('chain ' + pct(run.chain.recall) + ' end to end, ' + perFrame(run.chain.surviving_false_per_frame) + ' false per frame');
  return parts.join(' · ') || 'Classifier: no numbers yet.';
}

defineProps<{runs: RunSummary[]; selected: string | null}>();
const emit = defineEmits<{select: [name: string]}>();
</script>

<template>
  <div class="table-wrap">
    <table class="runs" id="runs">
      <thead>
        <tr><th>Run</th><th>Started from</th><th>State</th>
          <th class="num" title="Share of the test crops the reader reads right">Reader</th>
          <th class="num" title="The same on the crops a person confirmed">Reader, human</th>
          <th class="num" title="Share of the reviewed signs of the test the detector finds">Det. recall</th>
          <th class="num" title="Share of the detector's boxes that are a reviewed sign">Det. precision</th></tr>
      </thead>
      <tbody>
        <tr v-for="run in runs" :key="run.name" :class="{on: run.name === selected}" :data-run="run.name" @click="emit('select', run.name)">
          <td class="run-cell">
            <button type="button" class="run-name" :aria-pressed="run.name === selected" @click.stop="emit('select', run.name)">{{ run.name }}</button>
            <span class="run-sub">{{ runWhen(run) }}</span>
          </td>
          <td>{{ run.start.label }}<span v-if="modeText(run)" class="run-sub">{{ modeText(run) }}</span></td>
          <td><span class="pill" :class="RUN_STATE[run.state][1]">{{ RUN_STATE[run.state][0] }}</span><span class="run-sub">{{ stageText(run) }}</span></td>
          <td v-if="run.evaluated && run.kind === 'cutter'" colspan="4" class="run-cutter">{{ cutterLine(run) }}</td>
          <td v-else-if="run.evaluated && run.kind === 'usernet'" colspan="4" class="run-cutter">{{ usernetLine(run) }}</td>
          <td v-else-if="run.evaluated && run.kind === 'classifier'" colspan="4" class="run-classifier">{{ classifierLine(run) }}<span
            v-if="run.chain_old_reader" class="run-sub">old reader in the same chain: {{ pct(run.chain_old_reader.recall) }} end to end</span></td>
          <template v-else-if="run.evaluated">
            <td class="num">{{ pct(run.reader?.accuracy) }}<span v-if="run.reader" class="run-sub">{{ thousands(run.reader.samples) }} crops</span></td>
            <td class="num">{{ pct(run.reader?.human_accuracy) }}<span v-if="run.reader?.human_samples" class="run-sub">{{ thousands(run.reader.human_samples) }} crops</span></td>
            <td class="num">{{ pct(run.detector?.recall) }}<span v-if="run.detector" class="run-sub">{{ thousands(run.detector.found) }} of {{ thousands(run.detector.truth_boxes) }}</span></td>
            <td class="num">{{ pct(run.detector?.precision) }}<span v-if="run.detector" class="run-sub">{{ thousands(run.detector.false_boxes) }} false</span></td>
          </template>
          <td v-else colspan="4" class="run-none">{{ notEvaluatedText(run) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
