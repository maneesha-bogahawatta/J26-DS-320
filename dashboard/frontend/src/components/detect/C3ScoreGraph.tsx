import type { C3Job } from '../../api/c3Client';
import { scoreSegments } from '../../lib/c3Presentation';

export function C3ScoreGraph({ job, onSeek }: { job: C3Job; onSeek: (sourceTime: number) => void }) {
  const context = job.prediction?.source_context;
  if (!context || !job.scores?.length) return null;
  const [start, end] = context;
  return <div className="c3-notice" aria-label="C3 model score graph">
    <strong>Uncalibrated model scores</strong>
    <p>Blue: potential onset · Orange: active fight. Gaps mean no usable score. Click the graph to seek the recording.</p>
    <svg viewBox="0 0 920 210" role="img" aria-label="Onset and active-fight scores over original-video time"
      style={{ width: '100%', cursor: 'crosshair' }} onClick={event => {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width * 920;
        onSeek(start + Math.max(0, Math.min(1, (x - 42) / 836)) * (end - start));
      }}>
      {[0, .5, 1].map(value => <g key={value}>
        <line x1="42" x2="878" y1={170 - value * 140} y2={170 - value * 140} stroke="#526271" />
        <text x="5" y={175 - value * 140} fill="currentColor" fontSize="18">{value}</text>
      </g>)}
      {(['weak_onset_score', 'fight_score'] as const).map((field, h) =>
        scoreSegments(job.scores ?? [], field, start, end).map((points, i) =>
          <polyline key={`${field}-${i}`} points={points} fill="none" stroke={h ? '#f0ab4f' : '#60bbff'} strokeWidth="3" />))}
      <text x="42" y="202" fill="currentColor" fontSize="18">{start.toFixed(2)}s</text>
      <text x="790" y="202" fill="currentColor" fontSize="18">{end.toFixed(2)}s</text>
    </svg>
  </div>;
}
