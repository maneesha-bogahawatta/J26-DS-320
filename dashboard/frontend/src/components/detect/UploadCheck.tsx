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
import { c3Found, emptyVideoSummary } from '../../detect/recordedVideo';
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
import { C3RecordedResults } from './C3RecordedResults';
import { C3ScoreGraph } from './C3ScoreGraph';
import type { C3Job } from '../../api/c3Client';
import { annotatedContext, mediaOffset } from '../../lib/c3Presentation';

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
  const [recorded, setRecorded] = useState<C3Job | null>(null);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [failedMedia, setFailedMedia] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const pendingSourceSeek = useRef<number | null>(null);
  const completedJob = report?.c3?.status === 'completed' ? report.c3.job : undefined;
  const context = annotatedContext(completedJob);
  const annotated = phase === 'done' && context !== null && (showAnnotated || recorded !== null);
  const sourceStart = annotated ? context!.start : 0;
  const sourceCurrent = current + sourceStart;
  const playbackUrl = annotated ? c3MediaUrl(completedJob!.job_id) : url;

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
      setRecorded(null);
      setShowAnnotated(true);
      setFailedMedia(null);
      pendingSourceSeek.current = null;
      setCurrent(0);
      setDuration(0);
      setProgress({ fraction: 0, stage: 'Opening the video' });
      setPhase('checking');

      try {
        const completion: { report: VideoCheckReport | null } = { report: null };
        const result = await detector.analyzeVideo(chosen, {
          sourceId,
          mediaUrl: localUrl,
          onProgress: setProgress,
          onReport: value => { completion.report = value; setReport(value); },
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
        // Changing to the rendered artifact reloads media metadata. Seek only
        // after it is loaded; candidate timestamps remain in source-video time.
        if (result[0]) {
          const range = annotatedContext(completion.report?.c3?.job);
          const firstTime = result[0].startSeconds;
          if (range && firstTime >= range.start && firstTime <= range.end) pendingSourceSeek.current = firstTime;
          else {
            if (range) setShowAnnotated(false);
            if (videoRef.current) videoRef.current.currentTime = firstTime;
          }
        }
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
    setRecorded(null);
    setReport(null);
    setCurrent(0);
    setDuration(0);
    setFailedMedia(null);
    pendingSourceSeek.current = null;
  }, [url, found.length]);

  const seek = useCallback((t: number) => {
    const video = videoRef.current;
    if (annotated && context && (t < context.start || t > context.end) && !recorded) {
      // A teammate's event may be outside the C3 excerpt. Keep it available
      // on its original upload instead of drawing it at a false excerpt time.
      pendingSourceSeek.current = t;
      setShowAnnotated(false);
      return;
    }
    if (video) {
      video.pause();
      const offset = mediaOffset(t, sourceStart);
      if (video.readyState >= 1) video.currentTime = Math.min(video.duration || offset, offset);
      else pendingSourceSeek.current = t;
    }
  }, [annotated, context, recorded, sourceStart]);

  const pick = useCallback((f: Found) => {
    setSelectedId(f.event.event_id);
    seek(f.startSeconds);
  }, [seek]);

  const switchVideo = (nextAnnotated: boolean) => {
    if (nextAnnotated === annotated) return;
    pendingSourceSeek.current = sourceCurrent;
    setShowAnnotated(nextAnnotated);
  };

  const mediaLoaded = (seconds: number) => {
    setDuration(seconds);
    const video = videoRef.current;
    if (video && video.readyState >= 1 && pendingSourceSeek.current !== null) {
      video.currentTime = Math.min(seconds, mediaOffset(pendingSourceSeek.current, sourceStart));
      pendingSourceSeek.current = null;
    }
  };

  const boxes: StageBox[] = useMemo(() => {
    if (phase !== 'done' || annotated) return [];
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
  }, [found, current, phase, annotated]);

  const selected = found.find((f) => f.event.event_id === selectedId) ?? null;
  const openRecorded = (job: C3Job) => {
    const sourceStart = job.prediction?.source_context[0] ?? 0;
    const items = (job.events ?? []).map(event => c3Found(event, sourceStart));
    setRecorded(job); setFile(null); setUrl(c3MediaUrl(job.job_id)); setFound(items);
    setReport({ team: { status: 'completed', detectors: [] }, c3: { status: 'completed', job } });
    setSelectedId(items[0]?.event.event_id ?? null); setCurrent(0); setPhase('done');
    setShowAnnotated(true); setFailedMedia(null); setDuration(0); pendingSourceSeek.current = null;
  };

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
        {C3_ENABLED ? <C3RecordedResults onSelect={openRecorded} /> : null}
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
            {recorded?.source_name ?? file?.name}
          </p>
          <button type="button" className="btn btn--small" onClick={reset}>
            {phase === 'checking' ? 'Cancel' : 'Check another video'}
          </button>
        </div>

        {phase === 'done' && context ? <div className="c3-playback-options" role="group" aria-label="Video view">
          <button type="button" className={`btn btn--small${annotated ? ' btn--primary' : ''}`}
            aria-pressed={annotated} onClick={() => switchVideo(true)}>Skeleton video</button>
          {!recorded ? <button type="button" className={`btn btn--small${!annotated ? ' btn--primary' : ''}`}
            aria-pressed={!annotated} onClick={() => switchVideo(false)}>Original video</button> : null}
        </div> : null}
        <DetectStage
          videoRef={videoRef}
          src={playbackUrl}
          boxes={boxes}
          controls={phase === 'done'}
          onTimeUpdate={setCurrent}
          onDuration={mediaLoaded}
          onError={() => setFailedMedia(playbackUrl)}
        />
        {annotated ? <p className="muted-note">Skeleton overlay: green = observed joints; hollow amber = estimated joints; missing joints are not drawn. Low pose quality can prevent model scoring. Original-video range: {context!.start.toFixed(2)}–{context!.end.toFixed(2)} seconds.</p> : null}
        {playbackUrl && failedMedia === playbackUrl ? <p className="upload__error" role="alert">
          {annotated ? 'The annotated video could not be played. Use the Annotated video link below or switch to Original video.' : 'The original video could not be played.'}
        </p> : null}
        {recorded ? <div className="c3-notice"><strong>Recorded inference replay</strong>
          <p>Displaying authentic output computed at {recorded.created_at}. No new inference runs during replay.</p>
        </div> : null}
        {report?.c3?.job ? <C3ScoreGraph job={report.c3.job} onSeek={seek} /> : null}

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
            startSeconds={sourceStart}
            current={sourceCurrent}
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
          <p>{recorded ? 'Recorded C3 output only; team detectors do not run during replay.' : `Registered team detectors: ${report.team.status === 'failed' ? `failed — ${report.team.error}` : report.team.detectors.join(', ') || 'none registered'}.`}</p>
          {report.c3 ? <>
            <p><strong>C3 violence research: {report.c3.status === 'completed' ? 'completed' : 'incomplete'}</strong></p>
            {report.c3.error ? <p role="alert">{report.c3.error}</p> : null}
            {report.c3.status === 'failed' ? <p>No skeleton video was generated for this failed check.
              {report.c3.job ? <> <a href={c3MediaUrl(report.c3.job.job_id, 'inference.log')}>View inference error</a></> : null}
            </p> : null}
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
