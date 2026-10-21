/**
 * Loading, empty and error states.
 *
 * The brief calls these out explicitly: a missing clip, an empty queue or a
 * dead backend must each produce a clear message, not a blank panel and not a
 * crash.
 */

export function EmptyState({
  glyph = '—',
  title,
  detail,
  action,
}: {
  glyph?: string;
  title: string;
  detail?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="state">
      <div className="state__glyph" aria-hidden="true">
        {glyph}
      </div>
      <p className="state__title">{title}</p>
      {detail ? <p className="state__detail">{detail}</p> : null}
      {action}
    </div>
  );
}

export function ErrorState({ title, detail, onRetry }: { title: string; detail?: string; onRetry?: () => void }) {
  return (
    <div className="state state--error">
      <div className="state__glyph" aria-hidden="true">
        ⚠
      </div>
      <p className="state__title">{title}</p>
      {detail ? <p className="state__detail">{detail}</p> : null}
      {onRetry ? (
        <button type="button" className="btn btn--small" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

/** Placeholder rows while the first fetch is in flight. */
export function QueueSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading alerts">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{ padding: 'var(--sp-3)', borderBottom: '1px solid var(--line)' }}>
          <div className="skeleton" style={{ height: 11, width: `${55 + ((i * 7) % 30)}%`, marginBottom: 8 }} />
          <div className="skeleton" style={{ height: 9, width: '40%' }} />
        </div>
      ))}
    </div>
  );
}
