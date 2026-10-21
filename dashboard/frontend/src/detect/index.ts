/**
 * Picks the detection engine from .env, once.
 *
 *   VITE_DETECT_MODE=browser   practice detector in the page (default, no server)
 *   VITE_DETECT_MODE=http      send to backend_mock/detect.py at VITE_API_BASE_URL
 */

import type { DetectionEngine } from './engine';
import { createBrowserEngine } from './browserEngine';
import { createHttpEngine } from './httpEngine';

const mode = (import.meta.env.VITE_DETECT_MODE ?? 'browser').toLowerCase();
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export const detector: DetectionEngine = mode === 'http' ? createHttpEngine(baseUrl) : createBrowserEngine();

export type { DetectionEngine, Found, LiveTick, AnalyzeProgress, PathPoint } from './engine';
export { Cancelled } from './engine';
