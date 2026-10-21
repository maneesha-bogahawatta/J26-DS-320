/**
 * A video — an uploaded file or a live camera — with boxes drawn over it.
 *
 * Box colours follow the one colour rule: a box around something the system
 * has raised takes that alert's urgency colour. The plain "something is
 * moving" outline on the live camera is white, because it is not an alert.
 */

import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { BoundingBox } from '../../types/events';
import type { UrgencyKey } from '../../lib/urgency';
import { EMPTY_RECT, boxStyle, containRect } from '../../lib/geometry';
import type { Rect } from '../../lib/geometry';

export interface StageBox {
  key: string;
  box: BoundingBox;
  label: string;
  /** Urgency colour, or 'moving' for the neutral live outline. */
  tone: UrgencyKey | 'moving';
}

interface DetectStageProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  /** A file URL to play, or... */
  src?: string | null;
  /** ...a live camera stream. */
  stream?: MediaStream | null;
  boxes: StageBox[];
  controls?: boolean;
  onTimeUpdate?: (t: number) => void;
  onDuration?: (d: number) => void;
  onError?: () => void;
}

export function DetectStage({
  videoRef,
  src,
  stream,
  boxes,
  controls = false,
  onTimeUpdate,
  onDuration,
  onError,
}: DetectStageProps) {
  const stageRef = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<Rect>(EMPTY_RECT);

  const measure = useCallback(() => {
    const video = videoRef.current;
    const stage = stageRef.current;
    if (!video || !stage) return;
    const b = stage.getBoundingClientRect();
    setRect(containRect(b.width, b.height, video.videoWidth, video.videoHeight));
  }, [videoRef]);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [measure]);

  // A MediaStream cannot go in a src attribute; it has to be attached directly.
  useLayoutEffect(() => {
    const video = videoRef.current;
    if (!video || src) return;
    if (video.srcObject !== (stream ?? null)) {
      video.srcObject = stream ?? null;
      if (stream) void video.play().catch(() => undefined);
    }
  }, [stream, src, videoRef]);

  return (
    <div className="stage" ref={stageRef} data-live={String(Boolean(stream))}>
      <video
        ref={videoRef}
        className="stage__video"
        src={src ?? undefined}
        controls={controls}
        muted
        playsInline
        autoPlay={Boolean(stream)}
        preload="metadata"
        onLoadedMetadata={(e) => {
          measure();
          const d = e.currentTarget.duration;
          if (Number.isFinite(d)) onDuration?.(d);
        }}
        onDurationChange={(e) => {
          const d = e.currentTarget.duration;
          if (Number.isFinite(d)) onDuration?.(d);
        }}
        onResize={measure}
        onTimeUpdate={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        onSeeked={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        onError={onError}
      />
      {boxes.map((b) => (
        <div key={b.key} className="stage__box" data-tone={b.tone} style={boxStyle(b.box, rect)} aria-hidden="true">
          <span className="stage__tag">{b.label}</span>
        </div>
      ))}
    </div>
  );
}
