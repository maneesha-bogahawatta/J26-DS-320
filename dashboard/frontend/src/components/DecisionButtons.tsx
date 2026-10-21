/**
 * The four decisions, as four big coloured buttons.
 *
 * Each button is filled with its meaning's colour, carries a plain label and
 * one line saying what it does. An officer should never have to learn the
 * project's vocabulary — "escalate" is our word, "Send help" is theirs.
 *
 * Research notes that matter here, not just UI ones:
 *   - "Nothing wrong" does NOT delete the alert. Rejected alerts are the most
 *     valuable rows in the dataset: they are the hard negatives, and the
 *     false-alarm rate is one of the measures this project reports.
 *   - The note is where the "why" behind a decision is captured, which is the
 *     qualitative half of the operator study. It is optional, because an
 *     officer in a hurry must never be blocked by a form.
 *   - The shift ID is a role label (OP-01), never a name, so the console needs
 *     no login and the log holds no personal data.
 */

import { useEffect, useState } from 'react';
import type { IncidentEvent, OperatorAction } from '../types/events';
import { ACTION_TARGET_STATUS } from '../types/events';

const OPERATOR_ID_KEY = 'j26ds320.operator_id';

function loadOperatorId(): string {
  try {
    return localStorage.getItem(OPERATOR_ID_KEY) ?? 'OP-01';
  } catch {
    return 'OP-01';
  }
}

const DECISIONS: {
  action: OperatorAction;
  label: string;
  meaning: string;
  tone: 'now' | 'later' | 'closed' | 'done';
}[] = [
  {
    action: 'escalate',
    label: 'Send help',
    meaning: 'Something is wrong — get someone there',
    tone: 'now',
  },
  {
    action: 'acknowledge',
    label: "I've seen it",
    meaning: 'Watching it, nothing to do yet',
    tone: 'later',
  },
  {
    action: 'false_alarm',
    label: 'Nothing wrong',
    meaning: 'The system got this one wrong',
    tone: 'closed',
  },
  {
    action: 'resolve',
    label: 'All sorted',
    meaning: 'It was real and it has been handled',
    tone: 'done',
  },
];

interface DecisionButtonsProps {
  event: IncidentEvent;
  pending: boolean;
  onAction: (action: OperatorAction, operatorId: string, note: string | null) => void;
}

export function DecisionButtons({ event, pending, onAction }: DecisionButtonsProps) {
  const [note, setNote] = useState('');
  const [showNote, setShowNote] = useState(false);
  const [operatorId, setOperatorId] = useState(loadOperatorId);

  // A note belongs to one alert. Clear it when the officer moves on.
  useEffect(() => {
    setNote('');
    setShowNote(false);
  }, [event.event_id]);

  useEffect(() => {
    try {
      localStorage.setItem(OPERATOR_ID_KEY, operatorId);
    } catch {
      /* private browsing or blocked storage — the console still works */
    }
  }, [operatorId]);

  function fire(action: OperatorAction) {
    onAction(action, operatorId.trim() || 'OP-01', note.trim() ? note : null);
    setNote('');
    setShowNote(false);
  }

  return (
    <div className="decisions">
      <div className="decisions__grid">
        {DECISIONS.map(({ action, label, meaning, tone }) => {
          const already = event.status === ACTION_TARGET_STATUS[action];
          return (
            <button
              key={action}
              type="button"
              className="decide"
              data-tone={tone}
              disabled={pending || already}
              onClick={() => fire(action)}
            >
              <span className="decide__label">{label}</span>
              <span className="decide__meaning">{already ? 'Already marked this way' : meaning}</span>
            </button>
          );
        })}
      </div>

      {/* The note is folded away. An officer in a hurry presses a button and
          moves on; one with something to record opens this. */}
      {showNote ? (
        <div className="note">
          <label className="note__label" htmlFor="decision-note">
            Note (optional) — what you saw, or why you decided this
          </label>
          <textarea
            id="decision-note"
            className="note__input"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="e.g. Person got up by themselves and walked off."
            autoFocus
          />
          <p className="muted-note">
            The note is saved with whichever button you press next.
          </p>
        </div>
      ) : (
        <button type="button" className="link-button" onClick={() => setShowNote(true)}>
          + Add a note
        </button>
      )}

      <div className="shift-id">
        <label htmlFor="shift-id">Shift ID</label>
        <input
          id="shift-id"
          value={operatorId}
          onChange={(e) => setOperatorId(e.target.value)}
          maxLength={12}
        />
        <span>A shift label like OP-01, not your name.</span>
      </div>
    </div>
  );
}
