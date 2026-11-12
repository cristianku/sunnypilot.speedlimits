// "Download reviews (JSONL)": the confirmed frames as reviewed_frame lines, the same line the server stores
// (review_db.build_reviewed_frame). Offline it is the only way to keep the review; with the server, a backup.
import {countryOf} from './frames';
import {report, state} from './store';
import type {Detection, FrameRow} from './types';

/** One line of the export: the report row with the reviewer's decisions (review_db.build_reviewed_frame). */
interface ReviewedFrame extends FrameRow {
  reviewer: string | null | undefined;
  review_timestamp: string | null | undefined;
  kind: 'reviewed_frame';
  ground_truth: false;
  training_ready: false;
  review_state: 'reviewed';
  label_basis: 'human_review' | 'ai_review';
  source_proposals: Detection[] | undefined;
  detections: Detection[];
  country: string | null;
  users_complete?: true;
  reviewer_kind?: 'ai';
  review_channel?: string | undefined;
}

function say(text: string, ok: boolean): void {
  state.exportMessage = {text, ok};
}

export function exportJsonl(): void {
  const confirmed = report.rows.flatMap((row) => {
    const s = state.edits[row.frame_key];
    return s && s.review_state === 'reviewed' ? [{row, s}] : [];
  });
  if (confirmed.some(({s}) => !s.reviewer || !s.review_timestamp)) {
    say('Confirm the old reviews again to record their reviewer and date.', false);
    return;
  }
  const lines = confirmed.map(({row, s}) => {
    const line: ReviewedFrame = {
      ...row,
      reviewer: s.reviewer,
      review_timestamp: s.review_timestamp,
      site_group: s.site_group ?? row.site_group ?? null,
      encounter_id: s.encounter_id ?? row.encounter_id ?? null,
      country: countryOf(row, s),
      kind: 'reviewed_frame',
      ground_truth: false,
      training_ready: false,
      review_state: 'reviewed',
      label_basis: 'human_review',
      source_proposals: row.detections,
      detections: s.detections.map(({carried_from: _from, check_position: _check, ...d}) => ({
        ...d, review_state: 'reviewed', value_state: d.value_kph == null ? 'unknown' : 'reviewed',
      })),
    };
    // Only a ticked frame has it, as the server stores it.
    if (s.users_complete === true) line.users_complete = true;
    // A frame an AI confirmed keeps its provenance (label_basis, reviewer_kind, channel), as the server stores it.
    if (s.reviewer_kind === 'ai') {
      line.label_basis = 'ai_review';
      line.reviewer_kind = 'ai';
      line.review_channel = s.channel;
    }
    return JSON.stringify(line);
  });
  if (lines.length === 0) {
    say('Confirm at least one frame before exporting.', false);
    return;
  }
  const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], {type: 'application/x-ndjson'}));
  // A download needs a link of its own, never on the page.
  const link = document.createElement('a');
  link.href = url;
  link.download = 'annotations-' + report.runId + '.jsonl';
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  say(lines.length + (lines.length === 1 ? ' frame exported.' : ' frames exported.'), true);
}
