/**
 * Where to draw a box over a video.
 *
 * Detectors send NORMALISED coordinates (0..1 of the source frame), but a
 * <video> inside a flexible panel is letterboxed — the painted picture is
 * usually smaller than the element. `containRect` finds the rectangle the
 * picture actually occupies, so a box placed inside it lands on the right
 * pixels at any window size, for any clip shape, including portrait phone
 * video.
 */

import type { BoundingBox } from '../types/events';

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export const EMPTY_RECT: Rect = { left: 0, top: 0, width: 0, height: 0 };

/** Where an `object-fit: contain` video actually paints inside its element. */
export function containRect(elementW: number, elementH: number, intrinsicW: number, intrinsicH: number): Rect {
  if (!elementW || !elementH || !intrinsicW || !intrinsicH) {
    return { left: 0, top: 0, width: elementW, height: elementH };
  }
  const scale = Math.min(elementW / intrinsicW, elementH / intrinsicH);
  const width = intrinsicW * scale;
  const height = intrinsicH * scale;
  return { left: (elementW - width) / 2, top: (elementH - height) / 2, width, height };
}

/** CSS position for a normalised box inside a painted rectangle. */
export function boxStyle(box: BoundingBox, rect: Rect) {
  return {
    left: rect.left + box.x_min * rect.width,
    top: rect.top + box.y_min * rect.height,
    width: Math.max(2, (box.x_max - box.x_min) * rect.width),
    height: Math.max(2, (box.y_max - box.y_min) * rect.height),
  };
}

/** Overlap of two boxes as a fraction of the smaller one. 0 = apart, 1 = one inside the other. */
export function overlap(a: BoundingBox, b: BoundingBox): number {
  const w = Math.min(a.x_max, b.x_max) - Math.max(a.x_min, b.x_min);
  const h = Math.min(a.y_max, b.y_max) - Math.max(a.y_min, b.y_min);
  if (w <= 0 || h <= 0) return 0;
  const areaA = (a.x_max - a.x_min) * (a.y_max - a.y_min);
  const areaB = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  return (w * h) / Math.max(1e-6, Math.min(areaA, areaB));
}

/** 75.4 -> "1:15". Time within a video, the way a media player shows it. */
export function videoClock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '--:--';
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m % 60).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}
