// Starting an action of the control pages (a job in the background) and stopping a running job, each asked
// first in the confirm dialog when it is long, remote or disruptive.
import {messageOf, post} from '../common/api';
import type {ActionName, ActionOptions, ActionState, Job, PipelineAction, StartedJob, TrainingStart} from '../common/panel-types';
import {askConfirm, type ConfirmOptions} from '../composables/confirm';
import {toast} from '../composables/toast';
import {regionWords} from './mapillary';
import type {InjectionKey} from 'vue';

/** The resume() of the control app, for the job cards (provided by ControlApp.vue). */
export const RESUME_JOB: InjectionKey<(job: Job) => Promise<void>> = Symbol('resume-job');
/** "Try again" on the card of a failed job (provided by ControlApp.vue): whether it can start now, and start it with
 * the proposer it had. */
export const RETRY_JOB: InjectionKey<{can: (job: Job) => boolean; run: (job: Job) => void}> = Symbol('retry-job');
/** The actions whose failed job can simply run again: no option but the proposer (a training has Resume). */
export const RETRY_ACTIONS: ActionName[] = ['fetch', 'prepare', 'prepare_again', 'setup_vlm', 'dataset', 'measure_proposer',
  'networks_prepare', 'road_types', 'mapillary_locations', 'mapillary_windows', 'review_box'];

/** The question before an action; null: it starts at the click (preparing, importing the reviews, the road types). */
/** The proposer as the Prepare tab names it. */
function proposerName(proposer: string | undefined): string {
  return proposer === 'ours' ? 'our cutter and classifier' : proposer === 'qwen' ? 'Qwen3-VL' : 'the Vietnamese YOLO of Hugging Face';
}

function question(name: ActionName, state: ActionState, start: TrainingStart | null, options: ActionOptions | null): ConfirmOptions | null {
  if (name === 'train' || name === 'train_continue') {
    const from = start ? ' It starts from: ' + start.label + '.' : '';
    return {title: 'Train the whole pipeline?', action: 'Start training',
      text: 'The import, the split, then the detector and the reader on the GPU: it takes hours.' + from};
  }
  if (name === 'train_cutter') {
    const what = options?.families === 'speed' ? 'the speed signs only' : 'the speed signs and the other signs';
    const where = options?.region && options.region !== 'all' ? ' taken in ' + regionWords(options.region) : '';
    const share = options?.share && options.share !== '100' ? options.share + ' % of the' : 'the';
    const time = options?.mapillary === 'on' ? 'about two hours' : options?.mapillary === 'pretrain' ? 'about three hours'
      : 'about 80 minutes';
    const mapillary = options?.mapillary === 'on' ? ' The speed signs of ' + share + ' Mapillary windows' + where
      + ' join its training frames, at most as many as ours (Settings), not its measures.'
      : options?.mapillary === 'pretrain' ? ' It first learns on ' + share + ' Mapillary windows' + where
        + ' alone, then on our drives from there: two trainings, not its measures.' : '';
    return {title: 'Train the cutter?', action: 'Start training',
      text: 'Trains SignNet, the cutter the comma runs, then measures it on validation and test. It finds ' + what
        + ' in the 896 × 288 band at the top of the frame: ' + time + ' on the GPU. Its run is a new one, named by '
        + 'the time it starts.' + mapillary};
  }
  if (name === 'mapillary_windows') {
    return {title: 'Cut the Mapillary windows?', action: 'Cut',
      text: 'Reads the Mapillary photos on the TrueNAS share and cuts each speed sign into a window the size of the car’s '
        + 'band, for the cutter’s Mapillary switch. The CPU for a while, no GPU and no internet; the windows already cut '
        + 'are kept.'};
  }
  if (name === 'regression') {
    return {title: 'Run the regression?', action: 'Run',
      text: 'Runs the chosen models along the certified drives, only what is missing, on the GPU: a few minutes per model and '
        + 'drive. It trains nothing and changes nothing on the car.'};
  }
  if (name === 'train_brain') {
    const works = options?.works === 'off' ? ' Without the roadworks.' : '';
    const driver = options?.driver === 'on' ? ' With the driver’s inputs.' : '';
    const mean = options?.pool === 'mean' ? ' With the picture as a mean.' : '';
    return {title: 'Train the brain?', action: 'Start training',
      text: 'First it reads the signs of every drive with sunnypilot off and its logs read as the car does today, and our networks in use '
        + 'read the drives they have not read yet for the next limit (about 40 s a drive), then it trains three seeds on '
        + 'the GPU: about 15 minutes. Three new versions in Models; '
        + 'nothing goes to the car until you choose one and Publish.' + works + driver + mean};
  }
  if (name === 'train_decider') {
    return {title: 'Train applicability?', action: 'Start training',
      text: 'Trains three applicability networks on the GPU (the crop alone; with position and speed; with the scene) and compares them: '
        + 'about 15 minutes. Its run is a new one, named by the time it starts.'};
  }
  if (name === 'train_classifier') {
    const where = options?.region && options.region !== 'all' ? ' taken in ' + regionWords(options.region) : '';
    const share = options?.share && options.share !== '100' ? options.share + ' % of the' : 'The';
    const mapillary = options?.mapillary === 'on' ? ' ' + share + ' signs of Mapillary' + where
      + ' join its training crops, not its measures.'
      : options?.mapillary === 'pretrain' ? ' It first learns on ' + share.toLowerCase() + ' signs of Mapillary' + where
        + ' alone, then on yours from there: two trainings, not its measures.' : '';
    const from = /^classifier:(.+)\/classifier$/.exec(options?.start ?? '')?.[1];
    const start = from ? ' It starts from the classifier of ' + from + ', not from ImageNet.' : '';
    return {title: 'Train the classifier?', action: 'Start training',
      text: 'Mines the false boxes of the cutter in use, trains the classifier with “not a sign” on the GPU, then measures it '
        + 'alone and in the chain: it takes hours. Its run is a new one, named by the time it starts.' + mapillary + start};
  }
  if (name === 'networks_publish') {
    return {title: 'Publish ' + (options?.package ?? 'the package') + ' on Hugging Face?', action: 'Publish', danger: true,
      text: 'Uploads the cutter, the classifier and the applicability network of this package, their model card and the catalog to '
        + 'Hugging Face, public: anyone can download them, and a copy may stay even if you delete it later. No frame '
        + 'or dataset is uploaded.'};
  }
  if (name === 'setup_vlm') {
    return {title: 'Install Qwen3-VL?', action: 'Install',
      text: 'Downloads the model (17.5 GB, the files already copied here are kept), installs what it needs and tries it on the GPU. '
        + 'It can take an hour on a slow line.'};
  }
  if (name === 'measure_proposer') {
    return {title: 'Measure ' + proposerName(options?.proposer) + '?', action: 'Measure',
      text: 'It proposes the signs of the reviewed test frames, as Prepare would, and compares them with the boxes you kept. '
        + 'Nothing changes in the reviews or the drives. Our networks and Qwen3-VL use the GPU.'};
  }
  if (name === 'ai_review' && options?.save === 'yes') {
    return {title: 'Review and save with the AI?', action: 'Review and save', danger: true,
      text: 'What it saves waits for you: each frame goes into training only once you confirm it on the Signs side of its drive. '
        + 'Frames a person confirmed are never touched.'};
  }
  if (name === 'ai_review') {
    return {title: 'Try the AI reviewer?', action: 'Try',
      text: 'It suggests the boxes of ' + (options?.limit ? 'up to ' + options.limit + ' ' : 'the ')
        + 'frames still to review; the Signs side of the drive shows them on each frame. Nothing is saved.'};
  }
  if (name === 'measure_ai_reviewer') {
    return {title: 'Measure ' + (options?.provider ?? 'the AI reviewer') + '?', action: 'Measure',
      text: 'It asks the AI about up to ' + (options?.limit ?? '60') + ' frames a person reviewed and compares its '
        + 'answers with theirs. Nothing is saved; it takes minutes to an hour.'};
  }
  if (name === 'prepare_again') {
    return {title: 'Propose the signs again?', action: 'Propose again',
      text: 'The frames not confirmed yet get new proposals from ' + proposerName(options?.proposer)
        + ', on every drive. The frames already confirmed stay as they are. Each drive is read only until its new '
        + 'proposals are in.'};
  }
  if (name === 'fetch') {
    return {title: state.label + '?', action: 'Download',
      text: 'Copies the new drives from the comma into the video folder, segment by segment.'};
  }
  if (name === 'update') {
    return {title: 'Update the site?', action: 'Update',
      text: 'git pull, the libraries, then the site restarts: it is away for a few seconds.'};
  }
  if (name === 'road_labeler_train' && options?.dry === 'on') {
    return {title: 'Try the road labeler’s training?', action: 'Dry run',
      text: 'A couple of steps on the CPU with a tiny encoder, on the start of two drives: it checks the chain works, '
        + 'learns nothing and writes nothing outside its own run. A minute or two; the GPU is not used.'};
  }
  if (name === 'road_labeler_train') {
    return {title: 'Train the road labeler?', action: 'Start training',
      text: 'Reads every scanned drive and its pictures at high definition, then trains on the GPU on the corrections of '
        + 'Speed review and measures itself on validation. The first time it decodes the videos of every drive (about 13 s '
        + 'of CPU a minute of drive, an hour and a half for today’s drives); later runs reuse them. It refuses to start '
        + 'while another job has the GPU. A new run in the Road labeler card; nothing is written into Speed review.'};
  }
  if (name === 'road_labeler_predict' && options?.dry === 'on') {
    return {title: 'Try Prepare Speed review?', action: 'Dry run',
      text: 'The start of every drive on the CPU with ' + (options.labeler ?? 'this run') + ': its chances go into the '
        + 'run’s own folder, nothing is written into Speed review.'};
  }
  if (name === 'road_labeler_predict') {
    const marks = options?.marks === 'off'
      ? ' Chances only: nothing is written into Speed review.'
      : ' Where it is sure, every head that passed its gate writes into Speed review, and that is final: the kinds of '
        + 'road and the roadworks as marks of net/' + (options?.labeler ?? 'the run') + ', the road types over the map. '
        + 'It replaces what an older run wrote there, except the marks a person kept; a person’s marks, removals and '
        + 'checks always win. You correct only where it is wrong.';
    return {title: 'Prepare Speed review with ' + (options?.labeler ?? 'the road labeler') + '?', action: 'Prepare',
      text: 'Runs the road labeler along every scanned drive on the GPU and keeps its chances in '
        + 'derived/road_labeler.' + marks};
  }
  if (name === 'sign_tracks' && options?.dry === 'on') {
    return {title: 'Try the signs on every frame?', action: 'Dry run',
      text: 'The first 41 frames of the first minute of ' + (options.routes?.split(',')[0] ?? 'the first drive')
        + ' on the CPU, checked against the frames of Sign review: a minute or so. It goes into derived/sign_tracks/dry, '
        + 'what Speed review shows is not touched; the GPU is not used.'};
  }
  if (name === 'sign_tracks') {
    const drives = options?.routes ? options.routes.split(',').length : 0;
    return {title: 'Find the signs on every frame?', action: 'Start',
      text: 'Runs the cutter and the classifier in use on every frame of ' + (drives ? drives + (drives === 1 ? ' drive' : ' drives')
        : 'every drive with its video') + ' on the GPU, about 50 frames a second (some 9 hours for every drive): the boxes '
        + 'Speed review draws over the video. A drive already read by these networks is left as it is. It refuses to '
        + 'start while another job has the GPU.'};
  }
  if (name === 'box_fit_train' && options?.dry === 'on') {
    return {title: 'Count the boxes for Fit box?', action: 'Dry run',
      text: 'Counts the boxes people drew' + (options.refiner === 'off' ? '' : ' and those the refiner would learn from')
        + ', and the drives left out to measure: a few seconds on the CPU. Nothing is trained or written.'};
  }
  if (name === 'box_fit_train') {
    const refiner = options?.refiner !== 'off';
    return {title: refiner ? 'Train Fit box and its refiner?' : 'Train the rankers of Fit box?', action: 'Start training',
      text: 'On the CPU, the GPU is not used. First the two rankers (colour trees and edge networks) are measured on '
        + 'drives left out, then trained on every box people drew: about 15 minutes; they replace colour.npz and edges.npz.'
        + (refiner ? ' Then the box refiner, a small network, trains for about an hour and replaces refiner.onnx only '
          + 'if it beats the rankers on the drives left out.' : '')
        + ' The review page uses the new models after the site restarts. The models are files of the checkout: '
        + 'commit them, or Update the site stops on the local changes.'};
  }
  // send_to_car, try_on_comma, comma_restore: the Publish and Models tabs ask themselves (models.ts publishQuestion,
  // tryQuestion, restoreQuestion), naming the SignNet and the classifier in use.
  return null;
}

/** What stopping a job keeps, and what starting it again does: only a download and the install carry on from where
 * they stopped; a training of the whole pipeline has Resume; a measure starts over. */
const STOP_TEXT: Partial<Record<ActionName, string>> = {
  fetch: 'The drives already copied stay: downloading again carries on from there.',
  prepare: 'The drives already prepared stay. The one being prepared now starts again from its first picture next time.',
  prepare_again: 'The drives already done keep their new signs; the one at work keeps its old ones. Propose again '
    + 'starts over on every drive with frames not confirmed, those done now included.',
  measure_proposer: 'The measurement is lost: measuring again starts from the first frame.',
  measure_ai_reviewer: 'The measurement is lost: measuring again starts from the first frame.',
  setup_vlm: 'The files already downloaded stay: installing again carries on from there.',
  train: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_continue: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_resume: 'The steps already done stay: Resume runs the missing ones on the same run.',
  train_classifier: 'This training is lost: training again starts a new run from the beginning.',
  train_cutter: 'This training is lost: training again starts a new run from the beginning.',
  train_decider: 'The applicability networks already trained stay; training again starts a new run.',
  train_brain: 'The seeds already trained stay in Models; training again starts a new run.',
  ai_review: 'The frames already saved stay saved; the others are left to review.',
  networks_prepare: 'Nothing is kept: preparing again starts over.',
  networks_publish: 'What Hugging Face already received may stay there.',
  road_types: 'The drives done stay; the next run does them all again in a minute.',
  mapillary_locations: 'The photos already located stay: locating again carries on from there.',
  mapillary_windows: 'The windows already cut stay: cutting again carries on from there.',
  send_to_car: 'What was already published or pushed stays. Publishing again carries on: what is in place is skipped, '
    + 'and a build not finished runs then.',
  try_on_comma: 'Stopped before the copy, nothing on the comma changed. Stopped during the copy or the wait, the networks '
    + 'already on the comma stay until “Back to the build\'s networks”.',
  comma_restore: 'Trying again puts the build\'s networks back; nothing else on the comma is touched.',
  road_labeler_train: 'The pictures of the drives already read stay: training again reads only the others, then starts '
    + 'over on a new run.',
  road_labeler_predict: 'The drives done keep what it wrote; preparing again does every drive again.',
  sign_tracks: 'The drives done stay, and the minutes done of the drive at work: running again carries on from there.',
  box_fit_train: 'Models already written stay (the rankers once their step is over); the refiner is written only at '
    + 'its end. Training again starts over.',
};

/** run() and stop() for the buttons; refresh reads /api/panel/status again. */
export function usePanelActions(refresh: () => Promise<void>) {
  /** start: where a training starts (the selector of the Train tab); the server's default without it. options: the
   * cutter's size and signs. */
  async function run(name: ActionName, state: ActionState, start: TrainingStart | null = null,
    options: ActionOptions | null = null): Promise<void> {
    const ask = question(name, state, start, options);
    if (ask !== null && !await askConfirm(ask)) return;
    try {
      const job = await post<StartedJob>('/api/panel/run/' + name, start ? {start: start.key} : options ? {...options} : {});
      toast('Started: ' + job.label);
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  async function stop(job: Job): Promise<void> {
    const ok = await askConfirm({title: 'Stop “' + job.label + '”?', action: 'Stop', danger: true,
      text: STOP_TEXT[job.action] ?? 'It stops where it is now.'});
    if (!ok) return;
    try {
      await post('/api/panel/job/' + job.id + '/stop');
      toast('Stopped: ' + job.label);
    } catch (error) {
      toast(messageOf(error), true);
    }
    void refresh();
  }

  /** "Resume": the steps a stopped training still misses, on the same run (helpers/3_train.sh --resume). */
  async function resume(job: Job): Promise<void> {
    if (!job.run) return;
    const ok = await askConfirm({title: 'Resume “' + job.run + '”?', action: 'Resume',
      text: 'Only the steps still missing run again, on the same run: what is done stays.'});
    if (!ok) return;
    try {
      const started = await post<StartedJob>('/api/panel/run/train_resume', {run: job.run});
      toast('Started: ' + started.label);
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  /** "Try again": the same action, with the proposer it had (asked first as the action's own button asks). */
  async function retry(job: Job, state: ActionState): Promise<void> {
    await run(job.action, state, null, job.proposer ? {proposer: job.proposer} : null);
  }

  /** Train › Pipeline: the trainings chosen, one after the other, asked first. */
  async function startPipeline(steps: {action: PipelineAction; options: ActionOptions}[], text: string): Promise<void> {
    const ok = await askConfirm({title: 'Train ' + (steps.length === 1 ? 'this network' : 'these ' + steps.length + ' in a row') + '?',
      action: 'Start training', text});
    if (!ok) return;
    try {
      await post('/api/panel/pipeline', {steps});
      toast('Started: the pipeline');
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  /** "Stop after this one": the training at work goes on, the next ones do not start. */
  async function stopPipeline(): Promise<void> {
    const ok = await askConfirm({title: 'Stop the pipeline after this training?', action: 'Stop after this one', danger: true,
      text: 'The training at work goes on to its end; the next ones of the pipeline do not start. To stop it too, use its Stop.'});
    if (!ok) return;
    try {
      await post('/api/panel/pipeline/stop');
      toast('The pipeline stops after this training');
    } catch (error) {
      toast(messageOf(error), true);
    }
    void refresh();
  }

  return {run, stop, resume, retry, startPipeline, stopPipeline};
}
