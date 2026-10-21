import type { BoundingBox } from '../types/events';
import type { Found } from './engine';

/** How long either side of a found moment its box stays on screen during playback. */
const LEAD_S = 0.5;
const TRAIL_S = 1.5;

/** Where this found item's box should be drawn at time `t`, or null if not now. */
export function boxAt(found: Found, t: number): BoundingBox | null {
  if (found.path.length === 0) return null;
  if (t < found.startSeconds - LEAD_S || t > found.endSeconds + TRAIL_S) return null;
  let best = found.path[0];
  let bestGap = Math.abs(best.t - t);
  for (const p of found.path) {
    const gap = Math.abs(p.t - t);
    if (gap < bestGap) {
      best = p;
      bestGap = gap;
    }
  }
  return best.box;
}

/** "lobby camera 2.mp4" -> "lobby-camera-2" — a readable camera id for an uploaded file. */
export function slugFromFilename(name: string): string {
  const base = name.replace(/\.[^.]+$/, '');
  const slug = base
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32);
  return slug || 'video';
}
