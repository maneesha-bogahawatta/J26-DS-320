/** Presentation only: missing pose/score values never become zero predictions. */
export interface ScoreRow {
  t_s: number | string;
  weak_onset_score: number | string | null;
  fight_score: number | string | null;
}

export function finiteScore(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : null;
}

export function scoreSegments(rows: ScoreRow[], field: 'weak_onset_score' | 'fight_score', start: number, end: number): string[] {
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return [];
  const segments: string[] = []; let points: string[] = [];
  for (const row of rows) {
    const value = finiteScore(row[field]); const t = Number(row.t_s);
    if (value === null || !Number.isFinite(t) || t < start || t > end) {
      if (points.length) segments.push(points.join(' '));
      points = []; continue;
    }
    points.push(`${42 + (t - start) / (end - start) * 836},${170 - value * 140}`);
  }
  if (points.length) segments.push(points.join(' '));
  return segments;
}

export function mediaOffset(originalSeconds: number, sourceStart: number): number {
  return Number.isFinite(originalSeconds) && Number.isFinite(sourceStart) ? Math.max(0, originalSeconds - sourceStart) : 0;
}

export function booleanEvidence(value: unknown): boolean { return value === true || value === 'True' || value === 'true'; }

/** Scores, candidate policy and pose abstention are separate observations. */
export function headDiagnostics(job: {
  prediction?: { thresholds?: [number, number]; research_alert_heads_enabled?: [boolean, boolean] };
  scores?: Array<ScoreRow & { valid_pose?: boolean | string }>;
}) {
  return (['weak_onset_score', 'fight_score'] as const).map((field, index) => {
    const values = (job.scores ?? []).filter(row => row.valid_pose === undefined || booleanEvidence(row.valid_pose))
      .map(row => finiteScore(row[field])).filter((value): value is number => value !== null);
    return { field, label: index === 0 ? 'Potential-onset' : 'Active-fight',
      peak: values.length ? Math.max(...values) : null,
      threshold: finiteScore(job.prediction?.thresholds?.[index]),
      candidateEnabled: job.prediction?.research_alert_heads_enabled?.[index] ?? null };
  });
}
