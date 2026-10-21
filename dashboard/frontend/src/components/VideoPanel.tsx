/**
 * The video, with the detection highlight drawn over it.
 *
 * The hard part is the highlight box. Detectors send NORMALISED coordinates
 * (0..1 of the source frame), but a <video> inside a flexible panel is
 * letterboxed — the painted picture is usually smaller than the element. If
 * the box were positioned as a naive percentage of the element, it would
 * drift off the subject on any clip whose shape differs from the panel.
 *
 * So we measure: read the intrinsic size from the video, read the element
 * size from a ResizeObserver, compute the `object-fit: contain` rectangle,
 * and place the box inside that. The box then sits on the right pixels at any
 * window size.
 *
 * Three missing-media cases are handled separately, because they mean
 * different things to an officer:
 *   - no clip was ever saved (media_ref is null)
 *   - a clip was referenced but will not load
 *   - the clip plays
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { BoundingBox, IncidentEvent } from '../types/events';
import { resolveMediaUrl } from '../api';
import { boxStyle, containRect } from '../lib/geometry';
import type { Rect } from '../lib/geometry';
import { plainTrack } from '../lib/plainLanguage';
import { EmptyState } from './States';

function Highlight({ box, rect, label }: { box: BoundingBox; rect: Rect; label: string }) {
  return (
    <div className="highlight" aria-hidden="true" style={boxStyle(box, rect)}>
      <span className="highlight__tag">{label}</span>
    </div>
  );
}

export function VideoPanel({ event }: { event: IncidentEvent }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  const [showBox, setShowBox] = useState(true);
  const [mediaError, setMediaError] = useState(false);
  const [rect, setRect] = useState<Rect>({ left: 0, top: 0, width: 0, height: 0 });

  const mediaUrl = resolveMediaUrl(event.media_ref ?? null);

  /**
   * A sibling .webm, if the clip is an .mp4. Browsers without H.264 take this
   * one; everything else takes the mp4. Nothing breaks if the file is absent.
   */
  const webmUrl =
    mediaUrl && !mediaUrl.includes('/api/v1/c3/media/') && /\.mp4($|\?)/i.test(mediaUrl) ? mediaUrl.replace(/\.mp4($|\?)/i, '.webm$1') : null;

  const measure = useCallback(() => {
    const video = videoRef.current;
    const stage = stageRef.current;
    if (!video || !stage) return;
    const bounds = stage.getBoundingClientRect();
    setRect(containRect(bounds.width, bounds.height, video.videoWidth, video.videoHeight));
  }, []);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(() => measure());
    observer.observe(stage);
    return () => observer.disconnect();
  }, [measure]);

  // New alert: clear the error state and jump to the moment it happened.
  useEffect(() => {
    setMediaError(false);
    const video = videoRef.current;
    if (!video || !mediaUrl) return;

    const onLoaded = () => {
      measure();
      const offset = event.media_offset_seconds;
      if (typeof offset === 'number' && Number.isFinite(offset) && offset > 0) {
        try {
          video.currentTime = Math.min(offset, Math.max(0, video.duration - 0.05));
        } catch {
          /* seeking before metadata is ready — harmless, the clip still plays */
        }
      }
    };

    video.addEventListener('loadedmetadata', onLoaded);
    return () => video.removeEventListener('loadedmetadata', onLoaded);
  }, [mediaUrl, event.event_id, event.media_offset_seconds, measure]);

  if (!event.media_ref) {
    return (
      <div className="video">
        <div className="video__stage video__stage--empty">
          <EmptyState
            glyph="⊘"
            title="No video was saved"
            detail="Judge this one from what the camera saw, listed below."
          />
        </div>
      </div>
    );
  }

  if (mediaError) {
    return (
      <div className="video">
        <div className="video__stage video__stage--empty">
          <EmptyState
            glyph="⚠"
            title="The video will not play"
            detail="What the camera saw is still listed below. Tell the project team if this keeps happening."
          />
        </div>
      </div>
    );
  }

  return (
    <div className="video">
      <div className="video__stage" ref={stageRef}>
        {/*
          Two sources, not one. Some Linux builds of Chromium and Firefox ship
          without H.264 and refuse the .mp4 with no useful error. If a .webm
          sits beside it, the browser picks that instead and the demo works on
          any machine. A missing .webm is harmless — the browser just falls
          through to the mp4.
        */}
        <video
          ref={videoRef}
          className="video__el"
          key={mediaUrl ?? 'none'}
          controls
          muted
          playsInline
          preload="metadata"
          onError={() => setMediaError(true)}
          onLoadedData={measure}
        >
          {webmUrl ? <source src={webmUrl} type="video/webm" /> : null}
          {/* An uploaded file (blob:) can be any format, so no type is claimed for it. */}
          {mediaUrl && mediaUrl.startsWith('blob:') ? <source src={mediaUrl} /> : null}
          {mediaUrl && !mediaUrl.startsWith('blob:') ? <source src={mediaUrl} type="video/mp4" /> : null}
        </video>
        {showBox && event.bounding_box ? (
          <Highlight
            box={event.bounding_box}
            rect={rect}
            label={plainTrack(event.track_id) ?? 'Spotted here'}
          />
        ) : null}
      </div>

      {event.bounding_box ? (
        <label className="video__toggle">
          <input type="checkbox" checked={showBox} onChange={(e) => setShowBox(e.target.checked)} />
          Show what the system spotted
        </label>
      ) : null}
    </div>
  );
}
