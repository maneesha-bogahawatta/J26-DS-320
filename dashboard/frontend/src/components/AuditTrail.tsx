/**
 * Per-event audit trail.
 *
 * The proposal lists an "alert logging and audit trail module that timestamps,
 * records, and stores all system-generated alerts and operator actions for
 * post-incident review and accountability". This is the operator-facing half
 * of that: an append-only record of who decided what, when, and why.
 */

import type { AuditEntry } from '../types/events';
import { ACTION_LABELS } from '../lib/format';
import { plainClock, plainTimeAgo } from '../lib/plainLanguage';

export function AuditTrail({ entries }: { entries: AuditEntry[] }) {
  if (entries.length === 0) {
    return <p className="muted-note">Nobody has acted on this one yet.</p>;
  }

  const ordered = [...entries].sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime());

  return (
    <ul className="audit">
      {ordered.map((entry, i) => (
        <li className="audit__entry" key={`${entry.at}-${entry.action}-${i}`}>
          <span className="audit__pip" data-action={entry.action} aria-hidden="true" />
          <div>
            <div className="audit__action">
              {ACTION_LABELS[entry.action] ?? entry.action}
              {entry.operator_id ? <span className="audit__who"> by {entry.operator_id}</span> : null}
            </div>
            <div className="audit__when">
              {plainClock(entry.at)} · {plainTimeAgo(entry.at)}
            </div>
            {entry.note ? <div className="audit__note">{entry.note}</div> : null}
          </div>
        </li>
      ))}
    </ul>
  );
}
