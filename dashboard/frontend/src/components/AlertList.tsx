/**
 * The list of alerts — the officer's home screen.
 *
 * Three filter buttons, then cards. That is the entire interface.
 *
 * The filters are deliberately NOT the full set the system can do. An officer
 * needs "what's outstanding?" and "what happened earlier?", not a matrix of
 * component, camera, incident type, status and priority. The full filtering
 * still exists in the shift log, where someone is searching rather than
 * responding.
 */

import type { CameraSource, IncidentEvent } from '../types/events';
import { needsLook, sortByUrgency } from '../lib/urgency';
import { AlertCard } from './AlertCard';
import { EmptyState, QueueSkeleton } from './States';

/**
 * Four views, and every alert is in exactly one of the first three.
 *
 * The counts must never contradict the header. "Need a look" means the same
 * thing in both places — nobody has made a decision on it yet. An alert
 * somebody has acknowledged or escalated has been looked at; it is being
 * handled, which is a different question and gets its own view.
 */
export type ListFilter = 'new' | 'handling' | 'done' | 'all';

export const FILTERS: { key: ListFilter; label: string }[] = [
  { key: 'new', label: 'Need a look' },
  { key: 'handling', label: 'Being handled' },
  { key: 'done', label: 'Finished' },
  { key: 'all', label: 'Everything' },
];

export function applyListFilter(events: IncidentEvent[], filter: ListFilter): IncidentEvent[] {
  if (filter === 'new') return events.filter((e) => e.status === 'new');
  if (filter === 'handling')
    return events.filter((e) => e.status === 'acknowledged' || e.status === 'escalated');
  if (filter === 'done')
    return events.filter((e) => e.status === 'resolved' || e.status === 'false_positive');
  return events;
}

interface AlertListProps {
  events: IncidentEvent[];
  cameras: CameraSource[];
  filter: ListFilter;
  onFilterChange: (filter: ListFilter) => void;
  counts: Record<ListFilter, number>;
  selectedId: string | null;
  onOpen: (eventId: string) => void;
  loading: boolean;
}

export function AlertList({
  events,
  cameras,
  filter,
  onFilterChange,
  counts,
  selectedId,
  onOpen,
  loading,
}: AlertListProps) {
  const sorted = sortByUrgency(events);
  const unreviewed = events.filter(needsLook).length;

  return (
    <div className="list">
      <div className="list__filters" role="tablist" aria-label="Which alerts to show">
        {FILTERS.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            role="tab"
            className="filter"
            aria-selected={filter === key}
            onClick={() => onFilterChange(key)}
          >
            {label}
            <span className="filter__count">{counts[key]}</span>
          </button>
        ))}
      </div>

      <div className="list__scroll">
        {loading && events.length === 0 ? (
          <QueueSkeleton />
        ) : sorted.length === 0 ? (
          filter === 'new' ? (
            <EmptyState
              glyph="✓"
              title="All clear"
              detail="Nothing is waiting for you. New alerts appear here on their own."
            />
          ) : (
            <EmptyState glyph="○" title="Nothing here" detail="No alerts in this view." />
          )
        ) : (
          <>
            {filter === 'new' && unreviewed > 0 ? (
              <p className="list__lead">Most urgent first.</p>
            ) : null}

            <ul className="list__items">
              {sorted.map((event) => (
                <li key={event.event_id}>
                  <AlertCard
                    event={event}
                    cameras={cameras}
                    selected={event.event_id === selectedId}
                    onOpen={() => onOpen(event.event_id)}
                  />
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
