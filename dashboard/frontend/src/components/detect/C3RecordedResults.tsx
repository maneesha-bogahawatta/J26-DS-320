/** Genuine completed C3 jobs only; selecting one never runs inference. */
import { useEffect, useState } from 'react';
import { c3Bootstrap, c3Request } from '../../api/c3Client';
import type { C3Job } from '../../api/c3Client';

export function C3RecordedResults({ onSelect }: { onSelect: (job: C3Job) => void }) {
  const [jobs, setJobs] = useState<C3Job[]>([]);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    c3Bootstrap(controller.signal).then(body => setJobs(body.jobs.filter(job => job.status === 'completed')))
      .catch(error => { if (!controller.signal.aborted) setError(String(error)); });
    return () => controller.abort();
  }, []);
  async function open() {
    if (!selected) return;
    setBusy(true); setError('');
    try {
      const job = await c3Request<C3Job>(`/api/v1/c3/jobs/${encodeURIComponent(selected)}`);
      if (job.status !== 'completed' || !job.prediction || !job.scores?.length) throw new Error('Recorded inference artifacts are incomplete.');
      onSelect(job);
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }
  return <div className="c3-notice" aria-label="Recorded C3 inference results">
    <strong>Replay recorded C3 results</strong>
    <p>Authentic previously computed model output. Replaying a result runs no new detection or inference.</p>
    <label>Completed recording{' '}
      <select value={selected} onChange={e => setSelected(e.target.value)}>
        <option value="">Choose a recorded result</option>
        {jobs.map(job => <option key={job.job_id} value={job.job_id}>{job.source_name} · {job.created_at}</option>)}
      </select>
    </label>{' '}
    <button type="button" className="btn" disabled={!selected || busy} onClick={() => void open()}>{busy ? 'Opening…' : 'Replay recorded result'}</button>
    {!jobs.length ? <p>No completed C3 results are available in this server’s history.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
  </div>;
}
