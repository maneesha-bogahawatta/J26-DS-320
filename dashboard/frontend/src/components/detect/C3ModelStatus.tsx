import type { C3Job } from '../../api/c3Client';
import { headDiagnostics } from '../../lib/c3Presentation';

export function C3ModelStatus({ job }: { job: C3Job }) {
  return <div className="c3-head-status" aria-label="C3 scores and candidate policy">
    {headDiagnostics(job).map(head => <p key={head.field}>
      <strong>{head.label}:</strong> peak raw score {head.peak?.toFixed(3) ?? 'unavailable'};
      {' '}reference threshold {head.threshold?.toFixed(3) ?? 'not reported'}.
      {' '}{head.candidateEnabled === false ? 'Candidate generation is disabled by this checkpoint’s reliability policy; this is not a negative detection.'
        : head.candidateEnabled === true ? 'Candidate policy enabled; temporal confirmation and pose checks also apply.' : 'Candidate eligibility was not reported.'}
    </p>)}
    <p className="muted-note">These are uncalibrated research scores, not probabilities. No candidate does not establish absence of violence.</p>
  </div>;
}
