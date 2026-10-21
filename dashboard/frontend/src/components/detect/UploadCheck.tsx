/**
 * Check a recorded video.
 *
 * Three steps, one at a time:
 *   1. Choose — a big drop area. That is the only thing on screen.
 *   2. Checking — the video and a progress bar. Can be cancelled.
 *   3. Results — the video with a timeline of what was found, the list, and
 *      the selected item explained. Every item also goes into Alerts.
 *
 * With the practice detector the video never leaves the device: it is read
 * from a local file URL and examined in the page.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { IncidentEvent } from '../../types/events';
import { C3_ENABLED } from '../../api';
import type { VideoCheckReport } from '../../detect/engine';
import { emptyVideoSummary } from '../../detect/recordedVideo';
import { c3ExportUrl, c3MediaUrl } from '../../api/c3Client';
import { detector, Cancelled } from '../../detect';
import type { AnalyzeProgress, Found } from '../../detect';
import { boxAt, slugFromFilename } from '../../detect/playback';
import { describeIncident } from '../../lib/incidentTypes';
import { plainTrack } from '../../lib/plainLanguage';
import { urgencyOf } from '../../lib/urgency';
import { videoClock } from '../../lib/geometry';
import { AlertCard } from '../AlertCard';
import { DetectStage } from './DetectStage';
import type { StageBox } from './DetectStage';
import { FoundSummary } from './FoundSummary';
import { Timeline } from './Timeline';
import { C3ModelStatus } from './C3ModelStatus';

type Phase = 'choose' | 'checking' | 'done' | 'error';

interface UploadCheckProps {
  /** Called once per check with everything found, so it can go into Alerts. */
  onFound: (events: IncidentEvent[], source: { id: string; label: string }) => void;
  onOpenAlert: (eventId: string) => void;
}

export function UploadCheck({ onFound, onOpenAlert }: UploadCheckProps) {
  const [phase, setPhase] = useState<Phase>('choose');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState<AnalyzeProgress>({ fraction: 0, stage: '' });
  const [found, setFound] = useState<Found[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [report, setReport] = useState<VideoCheckReport | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cancel a running check if the screen goes away.
  useEffect(() => () => abortRef.current?.abort(), []);

  const start = useCallback(
    async (chosen: File) => {
      if (!chosen.type.startsWith('video/') && !/\.(mp4|webm|mov|m4v|mkv|avi|mpe?g|ogv)$/i.test(chosen.name)) {
        setError('That file is not a video. Choose an MP4, MOV or WebM file.');
        setPhase('error');
        return;
      }

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const localUrl = URL.createObjectURL(chosen);
      const sourceId = `upload-${slugFromFilename(chosen.name)}`;
      setFile(chosen);
      setUrl(localUrl);
      setFound([]);
      setSelectedId(null);
      setError(null);
      setReport(null);
      setCurrent(0);
      setProgress({ fraction: 0, stage: 'Opening the video' });
      setPhase('checking');

      try {
        const result = await detector.analyzeVideo(chosen, {
          sourceId,
          mediaUrl: localUrl,
          onProgress: setProgress,
          onReport: setReport,
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setFound(result);
        setSelectedId(result[0]?.event.event_id ?? null);
        setPhase('done');
        onFound(
          result.map((f) => f.event),
          { id: sourceId, label: `Uploaded — ${chosen.name}` },
        );
        if (result[0] && videoRef.current) videoRef.current.currentTime = result[0].startSeconds;
      } catch (err) {
        if (err instanceof Cancelled || controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : 'Something went wrong while checking the video.');
        setPhase('error');
      }
    },
    [onFound],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    // The URL is kept alive if anything was found: those alerts still play it.
    if (url && found.length === 0) URL.revokeObjectURL(url);
    setPhase('choose');
    setFile(null);
    setUrl(null);
    setFound([]);
    setSelectedId(null);
    setError(null);
  }, [url, found.length]);

  const pick = useCallback((f: Found) => {
    setSelectedId(f.event.event_id);
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.currentTime = f.startSeconds;
    }
  }, []);

  const seek = useCallback((t: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = t;
  }, []);

  const boxes: StageBox[] = useMemo(() => {
    if (phase !== 'done') return [];
    const out: StageBox[] = [];
    for (const f of found) {
      const box = boxAt(f, current);
      if (!box) continue;
      out.push({
        key: f.event.event_id,
        box,
        label: plainTrack(f.event.track_id) ?? describeIncident(f.event.incident_type, f.event.component_id).label,
        tone: urgencyOf(f.event).key,
      });
    }
    return out;
  }, [found, current, phase]);

  const selected = found.find((f) => f.event.event_id === selectedId) ?? null;

  /* ------------------------------------------------------------ step 1 */

  if (phase === 'choose' || (phase === 'error' && !url)) {
    return (
      <div className="upload">
        <div
          className="drop"
          data-dragging={String(dragging)}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) void start(f);
          }}
        >
          <span className="drop__icon" aria-hidden="true">
            ⇪
          </span>
          <p className="drop__title">Drop a video here</p>
          <p className="drop__or">or</p>
          <button type="button" className="btn btn--primary btn--big" onClick={() => inputRef.current?.click()}>
            Choose a video
          </button>
          <p className="drop__hint">MP4, MOV or WebM. Phone recordings work.</p>
          {C3_ENABLED ? <p className="drop__hint">Includes C3 temporal violence research and registered team detectors. C3 checks up to the first 60 seconds; inspect pose coverage before interpreting results.</p> : null}
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="visually-hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = '';
              if (f) void start(f);
            }}
          />
        </div>
        {error ? (
          <p className="upload__error" role="alert">
            {error}
          </p>
        ) : null}
      </div>
    );
  }

  /* ------------------------------------------------------- steps 2 & 3 */

  const pct = progress.fraction === null ? null : Math.round(progress.fraction * 100);

  return (
    <div className="check" data-phase={phase}>
      <div className="check__main">
        <div className="check__titlebar">
          <p className="check__file" title={file?.name}>
            {file?.name}
          </p>
          <button type="button" className="btn btn--small" onClick={reset}>
            {phase === 'checking' ? 'Cancel' : 'Check another video'}
          </button>
        </div>

        <DetectStage
          videoRef={videoRef}
          src={url}
          boxes={boxes}
          controls={phase === 'done'}
          onTimeUpdate={setCurrent}
          onDuration={setDuration}
        />

        {phase === 'checking' ? (
          <div className="progress" role="status" aria-live="polite">
            <div className="progress__bar" data-indeterminate={String(pct === null)}>
              <span style={{ width: pct === null ? undefined : `${pct}%` }} />
            </div>
            <p className="progress__text">
              {progress.stage}
              {pct !== null ? ` — ${pct}%` : '…'}
            </p>
            {C3_ENABLED ? <p className="muted-note">C3 pose extraction can be slower than playback. Cancel stops waiting here; an already submitted job continues on the server.</p> : null}
          </div>
        ) : null}

        {phase === 'done' ? (
          <Timeline
            duration={duration}
            current={current}
            found={found}
            selectedId={selectedId}
            onPick={pick}
            onSeek={seek}
          />
        ) : null}

        {phase === 'error' ? (
          <p className="upload__error" role="alert">
            {error}
          </p>
        ) : null}

        {phase === 'done' && selected ? (
          <FoundSummary
            found={selected}
            when={`At ${videoClock(selected.startSeconds)} in the video`}
            onOpenAlert={onOpenAlert}
          />
        ) : null}
      </div>

      <aside className="check__side" aria-label="What was found">
        {report ? <div className="c3-notice" aria-label="Models checked">
          <strong>Models checked</strong>
          <p>Registered team detectors: {report.team.status === 'failed' ? `failed — ${report.team.error}` : report.team.detectors.join(', ') || 'none registered'}.</p>
          {report.c3 ? <>
            <p><strong>C3 violence research: {report.c3.status === 'completed' ? 'completed' : 'incomplete'}</strong></p>
            {report.c3.error ? <p role="alert">{report.c3.error}</p> : null}
            {report.c3.job?.prediction ? <>
              <p>Checked original-video seconds {report.c3.job.prediction.source_context[0].toFixed(2)}–{report.c3.job.prediction.source_context[1].toFixed(2)}. Usable pose decisions: {report.c3.job.prediction.timing.valid_decisions} / {report.c3.job.prediction.timing.decisions}.</p>
              <p>Weak-supervision scores are uncalibrated. Active-fight scores and potential-onset candidates are separate; a high score does not confirm violence.</p>
              <C3ModelStatus job={report.c3.job} />
              <p><a href={c3MediaUrl(report.c3.job.job_id)}>Annotated video</a> · <a href={c3ExportUrl(report.c3.job.job_id)}>Export C3 run</a></p>
            </> : null}
          </> : <p>C3 violence inference was not run in this configuration.</p>}
        </div> : null}
        {phase === 'checking' ? (
          <p className="check__side-lead">Results will appear here when the check is finished.</p>
        ) : null}

        {phase === 'done' && found.length === 0 ? (
          <div className="c3-notice">
            <p className="allclear__title">{emptyVideoSummary(report, C3_ENABLED)}</p>
            <p className="allclear__detail">
              No candidate was returned by the checks above. Missing poses, incomplete checks or low scores do not prove that the video contains no violence. Review the recording and the checked time range.
            </p>
          </div>
        ) : null}

        {phase === 'done' && found.length > 0 ? (
          <>
            <h2 className="check__side-title">
              Found {found.length} {found.length === 1 ? 'thing' : 'things'} worth a look
            </h2>
            <ul className="list__items">
              {found.map((f) => (
                <li key={f.event.event_id}>
                  <AlertCard
                    event={f.event}
                    cameras={[]}
                    selected={f.event.event_id === selectedId}
                    onOpen={() => pick(f)}
                    where={`At ${videoClock(f.startSeconds)} in the video`}
                  />
                </li>
              ))}
            </ul>
            <p className="check__side-note">Everything found is also in Alerts, where you can record a decision.</p>
          </>
        ) : null}
      </aside>
    </div>
  );
}
