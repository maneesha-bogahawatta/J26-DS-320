/**
 * Low-level motion measurement for the practice detector.
 *
 * Every frame is shrunk to a small greyscale picture and cut into a grid of
 * cells. A cell is "active" when enough of its pixels changed. The biggest
 * connected group of active cells is the thing that moved.
 *
 * Working on a 320x180 picture and a 64x36 grid is what lets this run in a
 * phone browser four times a second. It is also why it is a PRACTICE
 * detector: it sees that something moved and roughly where, not what it was.
 */

import type { BoundingBox } from '../types/events';

export const GRID_W = 64;
export const GRID_H = 36;
const CELL_PX = 5;
export const SAMPLE_W = GRID_W * CELL_PX; // 320
export const SAMPLE_H = GRID_H * CELL_PX; // 180
export const CELL_COUNT = GRID_W * GRID_H;

/** A pixel counts as changed above this difference in grey level (0..255). */
const PIXEL_DIFF = 28;
/** A cell counts as active when this fraction of its pixels changed. */
const CELL_FRACTION = 0.22;

export interface Region {
  box: BoundingBox;
  /** Number of grid cells in the region. */
  cells: number;
  /** Fraction of the whole picture the region covers, 0..1. */
  area: number;
}

/** Draws video frames onto a small canvas and returns them as greyscale. */
export class FrameSampler {
  private ctx: CanvasRenderingContext2D;

  constructor() {
    const canvas = document.createElement('canvas');
    canvas.width = SAMPLE_W;
    canvas.height = SAMPLE_H;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('This browser cannot read video frames (no 2D canvas).');
    this.ctx = ctx;
  }

  grab(source: CanvasImageSource): Uint8Array {
    this.ctx.drawImage(source, 0, 0, SAMPLE_W, SAMPLE_H);
    const rgba = this.ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H).data;
    const grey = new Uint8Array(SAMPLE_W * SAMPLE_H);
    for (let i = 0, p = 0; i < grey.length; i += 1, p += 4) {
      // Rec. 601 luma, integer form.
      grey[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
    }
    return grey;
  }
}

/**
 * Pixels this bright are blown out — sunlit windows, reflections, lamps. Their
 * brightness swings with the camera's exposure, not with anything happening,
 * so changes there are not trusted.
 */
const BLOWN_OUT = 238;

/** Which cells differ between two greyscale pictures. */
export function activeCells(a: ArrayLike<number>, b: ArrayLike<number>, pixelDiff = PIXEL_DIFF): Uint8Array {
  const counts = new Uint16Array(CELL_COUNT);
  for (let y = 0; y < SAMPLE_H; y += 1) {
    const row = y * SAMPLE_W;
    const cellRow = Math.floor(y / CELL_PX) * GRID_W;
    for (let x = 0; x < SAMPLE_W; x += 1) {
      const i = row + x;
      if (a[i] >= BLOWN_OUT && b[i] >= BLOWN_OUT - 40) continue;
      if (b[i] >= BLOWN_OUT && a[i] >= BLOWN_OUT - 40) continue;
      if (Math.abs(a[i] - b[i]) > pixelDiff) counts[cellRow + Math.floor(x / CELL_PX)] += 1;
    }
  }
  const threshold = CELL_PX * CELL_PX * CELL_FRACTION;
  const cells = new Uint8Array(CELL_COUNT);
  for (let c = 0; c < CELL_COUNT; c += 1) cells[c] = counts[c] >= threshold ? 1 : 0;
  return cells;
}

export function countCells(cells: Uint8Array): number {
  let n = 0;
  for (let c = 0; c < cells.length; c += 1) n += cells[c];
  return n;
}

/** Grow every active cell by one in each direction. */
export function dilate(cells: Uint8Array): Uint8Array {
  const out = new Uint8Array(CELL_COUNT);
  for (let y = 0; y < GRID_H; y += 1) {
    for (let x = 0; x < GRID_W; x += 1) {
      if (!cells[y * GRID_W + x]) continue;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < GRID_W && ny < GRID_H) out[ny * GRID_W + nx] = 1;
        }
      }
    }
  }
  return out;
}

/** a AND NOT b */
export function subtract(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(CELL_COUNT);
  for (let c = 0; c < CELL_COUNT; c += 1) out[c] = a[c] && !b[c] ? 1 : 0;
  return out;
}

/**
 * The biggest group of touching active cells, as a normalised box.
 *
 * Taking only the biggest group is the noise filter: compression flicker and
 * burned-in timestamps make small scattered groups, a person makes one big one.
 * Diagonal neighbours count as touching, so a thin limb does not split a body.
 */
export function largestRegion(cells: Uint8Array, minCells: number): Region | null {
  const seen = new Uint8Array(CELL_COUNT);
  const stack: number[] = [];
  let best: Region | null = null;

  for (let start = 0; start < CELL_COUNT; start += 1) {
    if (!cells[start] || seen[start]) continue;

    let n = 0;
    let minX = GRID_W;
    let minY = GRID_H;
    let maxX = -1;
    let maxY = -1;
    stack.push(start);
    seen[start] = 1;

    while (stack.length > 0) {
      const c = stack.pop() as number;
      const x = c % GRID_W;
      const y = (c - x) / GRID_W;
      n += 1;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H) continue;
          const nc = ny * GRID_W + nx;
          if (cells[nc] && !seen[nc]) {
            seen[nc] = 1;
            stack.push(nc);
          }
        }
      }
    }

    if (n >= minCells && (!best || n > best.cells)) {
      best = {
        cells: n,
        area: n / CELL_COUNT,
        box: {
          x_min: minX / GRID_W,
          y_min: minY / GRID_H,
          x_max: (maxX + 1) / GRID_W,
          y_max: (maxY + 1) / GRID_H,
        },
      };
    }
  }

  return best;
}

/**
 * A slowly-adapting picture of what the scene looks like when nothing is
 * happening. Comparing against it finds things that CHANGED AND STAYED — an
 * item put down, a person who went down and did not get up — which frame-to-
 * frame differences cannot see, because a still thing produces no motion.
 */
export class Background {
  private model: Float32Array;

  constructor(first: Uint8Array) {
    this.model = Float32Array.from(first);
  }

  reset(frame: Uint8Array): void {
    this.model = Float32Array.from(frame);
  }

  /**
   * Blend the new frame in, except where things are moving. `tau` is how many
   * seconds it takes the background to mostly absorb a change; `dt` is the
   * time since the last update.
   */
  update(frame: Uint8Array, moving: Uint8Array, dt: number, tau: number): void {
    const alpha = 1 - Math.exp(-Math.max(0, dt) / tau);
    for (let y = 0; y < SAMPLE_H; y += 1) {
      const row = y * SAMPLE_W;
      const cellRow = Math.floor(y / CELL_PX) * GRID_W;
      for (let x = 0; x < SAMPLE_W; x += 1) {
        if (moving[cellRow + Math.floor(x / CELL_PX)]) continue;
        const i = row + x;
        this.model[i] += (frame[i] - this.model[i]) * alpha;
      }
    }
  }

  changedCells(frame: Uint8Array): Uint8Array {
    return activeCells(frame, this.model, 34);
  }

  get pixels(): ArrayLike<number> {
    return this.model;
  }

  /** Accept the current picture inside a box as the new normal. */
  absorbBox(frame: Uint8Array, box: BoundingBox): void {
    const [x0, y0, x1, y1] = boxToPixels(box);
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) this.model[y * SAMPLE_W + x] = frame[y * SAMPLE_W + x];
    }
  }

  /** Accept the current picture in these squares as the new normal. */
  absorbCells(frame: Uint8Array, cells: Uint8Array): void {
    for (let y = 0; y < SAMPLE_H; y += 1) {
      const row = y * SAMPLE_W;
      const cellRow = Math.floor(y / CELL_PX) * GRID_W;
      for (let x = 0; x < SAMPLE_W; x += 1) {
        if (cells[cellRow + Math.floor(x / CELL_PX)]) this.model[row + x] = frame[row + x];
      }
    }
  }
}

/**
 * Squares that are only "new" because the BACKGROUND is out of date — the spot
 * a person or object was in when the background was first taken, and has
 * since left. These are called ghosts.
 *
 * The test: a ghost square now looks like the floor around it (it IS the
 * floor; the background is what is wrong). A real new object stands out from
 * its surroundings. Comparing each square's brightness with nearby squares
 * that have not changed separates the two even when they touch — which they
 * usually do, because a bag is put down right where its owner was standing.
 */
export function ghostCells(changed: Uint8Array, exclude: Uint8Array, frame: Uint8Array): Uint8Array {
  const means = cellMeans(frame);
  const ghosts = new Uint8Array(CELL_COUNT);
  const R = 5;
  for (let y = 0; y < GRID_H; y += 1) {
    for (let x = 0; x < GRID_W; x += 1) {
      const c = y * GRID_W + x;
      if (!changed[c]) continue;
      let sum = 0;
      let n = 0;
      for (let dy = -R; dy <= R; dy += 1) {
        for (let dx = -R; dx <= R; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H) continue;
          const nc = ny * GRID_W + nx;
          if (changed[nc] || exclude[nc]) continue;
          sum += means[nc];
          n += 1;
        }
      }
      if (n >= 3 && Math.abs(means[c] - sum / n) < GHOST_SIMILARITY) ghosts[c] = 1;
    }
  }
  return ghosts;
}

/** A square within this many grey levels of its surroundings looks like them. */
const GHOST_SIMILARITY = 14;

function cellMeans(frame: ArrayLike<number>): Float32Array {
  const sums = new Float32Array(CELL_COUNT);
  for (let y = 0; y < SAMPLE_H; y += 1) {
    const row = y * SAMPLE_W;
    const cellRow = Math.floor(y / CELL_PX) * GRID_W;
    for (let x = 0; x < SAMPLE_W; x += 1) sums[cellRow + Math.floor(x / CELL_PX)] += frame[row + x];
  }
  for (let c = 0; c < CELL_COUNT; c += 1) sums[c] /= CELL_PX * CELL_PX;
  return sums;
}

function boxToPixels(box: BoundingBox): [number, number, number, number] {
  return [
    Math.max(0, Math.floor(box.x_min * SAMPLE_W)),
    Math.max(0, Math.floor(box.y_min * SAMPLE_H)),
    Math.min(SAMPLE_W, Math.ceil(box.x_max * SAMPLE_W)),
    Math.min(SAMPLE_H, Math.ceil(box.y_max * SAMPLE_H)),
  ];
}

/**
 * Is the change inside this box only a change of LIGHT?
 *
 * When the sun comes out, a lamp flickers or the camera adjusts its exposure,
 * the picture gets brighter or darker but keeps the same pattern — the floor
 * tiles are still the floor tiles. An object put down replaces the pattern
 * with its own shape. So: if the two pictures are strongly correlated, or both
 * are featureless, it is light, not a thing.
 */
export function lightStats(background: ArrayLike<number>, frame: ArrayLike<number>, box: BoundingBox) {
  const [x0, y0, x1, y1] = boxToPixels(box);
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const i = y * SAMPLE_W + x;
      sa += background[i];
      sb += frame[i];
      n += 1;
    }
  }
  if (n < 4) return { n, meanBefore: 0, meanNow: 0, stdBefore: 0, stdNow: 0, correlation: 1 };
  const ma = sa / n;
  const mb = sb / n;
  let va = 0;
  let vb = 0;
  let cov = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const i = y * SAMPLE_W + x;
      const da = background[i] - ma;
      const db = frame[i] - mb;
      va += da * da;
      vb += db * db;
      cov += da * db;
    }
  }
  const stdA = Math.sqrt(va / n);
  const stdB = Math.sqrt(vb / n);
  const correlation = stdA > 1e-3 && stdB > 1e-3 ? cov / n / (stdA * stdB) : 0;
  return { n, meanBefore: ma, meanNow: mb, stdBefore: stdA, stdNow: stdB, correlation };
}
