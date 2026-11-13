// The Review with AI at work (panel.Panel.ai_live), shared by the control pages: ControlApp.vue sets it from the full
// status and from the light poll of /api/panel/system; Speed review marks its drive in the list with it.
import {shallowRef} from 'vue';
import type {AiLive} from '../common/panel-types';

export const aiLive = shallowRef<AiLive | null>(null);
