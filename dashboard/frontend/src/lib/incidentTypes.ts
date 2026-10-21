/**
 * Registry of the incident types the four components produce.
 *
 * IMPORTANT: this registry is for PRESENTATION ONLY. The dashboard must still
 * render an event whose `incident_type` is not listed here — see
 * `describeIncident()`, which falls back to a readable label instead of
 * failing. That is what lets Members 1-3 ship a new detector without waiting
 * for a dashboard change.
 *
 * Wording rule: every label is cautious. CCTV evidence does not prove a fall,
 * an owner, a crime or an intent. Labels say "Potential", "Possible" or
 * "Unusual" — never "Confirmed", "Detected crime" or "Medical emergency".
 */

import type { ComponentId } from '../types/events';

export interface ComponentMeta {
  id: ComponentId;
  /** Short tag used in commit messages and badges, e.g. "C1". */
  tag: string;
  /** Full component name from the research proposal. */
  name: string;
  /** Owning member, filled in from the onboarding guide. */
  owner: string;
  /** CSS custom property holding this component's accent colour. */
  colorVar: string;
}

export const COMPONENTS: Record<ComponentId, ComponentMeta> = {
  component_1: {
    id: 'component_1',
    tag: 'C1',
    name: 'Spatial Anomaly Detection',
    owner: 'Member 1',
    colorVar: '--c1',
  },
  component_2: {
    id: 'component_2',
    tag: 'C2',
    name: 'Predictive Loitering Detection',
    owner: 'Member 2',
    colorVar: '--c2',
  },
  component_3: {
    id: 'component_3',
    tag: 'C3',
    name: 'Aggression & Violence Recognition',
    owner: 'Member 3',
    colorVar: '--c3',
  },
  component_4: {
    id: 'component_4',
    tag: 'C4',
    name: 'Fall & Abandoned-Object Detection',
    owner: 'Bogahawatta M.O',
    colorVar: '--c4',
  },
};

export const COMPONENT_IDS = Object.keys(COMPONENTS) as ComponentId[];

export interface IncidentMeta {
  type: string;
  /** Cautious label shown in the queue and detail header. */
  label: string;
  /**
   * What to check in the video, in plain words. This is written for a security
   * officer, not an engineer — it names the thing that most often makes this
   * alert a false alarm, so they know what to rule out.
   */
  operatorHint: string;
  /**
   * Suggested first step. Always defers to the site's own procedure — the
   * system advises, it does not instruct, and it cannot see everything the
   * officer can.
   */
  whatToDo: string;
  /** Which component normally raises this. Used only for the legend. */
  component: ComponentId;
  /** Glyph shown in the queue. Deliberately abstract — not a status symbol. */
  glyph: string;
}

export const INCIDENT_TYPES: Record<string, IncidentMeta> = {
  // ---- C1 — Spatial Anomaly Detection ----
  tailgating: {
    type: 'tailgating',
    label: 'Someone may have followed another person in',
    operatorHint:
      'Two people went through a controlled door close together. The system cannot see badges — the second person may well have their own.',
    whatToDo: 'Check the door on the video. If the second person had no badge, follow your access procedure.',
    component: 'component_1',
    glyph: '⇥',
  },
  restricted_zone_intrusion: {
    type: 'restricted_zone_intrusion',
    label: 'Someone entered a restricted area',
    operatorHint:
      'A person went into an area marked as restricted. Staff and contractors who are allowed in look exactly the same to the camera.',
    whatToDo: 'Check whether the person is expected there — cleaning, maintenance and deliveries often are.',
    component: 'component_1',
    glyph: '⊘',
  },
  unauthorized_proximity: {
    type: 'unauthorized_proximity',
    label: 'Two people unusually close',
    operatorHint:
      'Two people stayed closer together, for longer, than is normal in this spot. Friends talking look the same.',
    whatToDo: 'Watch the clip. Look at whether either person seems uncomfortable.',
    component: 'component_1',
    glyph: '⇄',
  },
  zone_activity: {
    type: 'zone_activity',
    label: 'Movement spotted',
    operatorHint:
      'Something moved in view for a while. On its own this is usually ordinary — people walking past, a door, a shadow.',
    whatToDo: 'Glance at the clip. Only act if the movement is somewhere it should not be.',
    component: 'component_1',
    glyph: '◌',
  },
  crowding: {
    type: 'crowding',
    label: 'Crowd building up quickly',
    operatorHint:
      'Far more people than usual gathered here in a short time. Shift changes and queues do this too.',
    whatToDo: 'Check for a blockage or a queue forming. Watch for crush points near doors.',
    component: 'component_1',
    glyph: '⣿',
  },

  // ---- C2 — Predictive Loitering Detection ----
  loitering: {
    type: 'loitering',
    label: 'Someone hanging around',
    operatorHint:
      'A person has stayed in one spot and moved about without clear purpose. Waiting for someone looks very similar.',
    whatToDo: 'Watch for a minute. Most people are simply waiting — look at whether they keep checking the area.',
    component: 'component_2',
    glyph: '◷',
  },

  // ---- C3 — Aggression & Violence Recognition ----
  aggression: {
    type: 'aggression',
    label: 'Possible aggressive behaviour',
    operatorHint:
      'Someone is moving in a way that can come before a confrontation. Lively conversation and gesturing look similar.',
    whatToDo: 'Watch the clip now. If it looks like it may escalate, send someone before it does.',
    component: 'component_3',
    glyph: '⚡',
  },
  violence: {
    type: 'violence',
    label: 'Possible fight',
    operatorHint:
      'Two or more people are moving in a way that looks like a physical fight. Horseplay can look the same.',
    whatToDo: 'Watch the clip immediately. If it is a real fight, follow your response procedure.',
    component: 'component_3',
    glyph: '✸',
  },
  concealment: {
    type: 'concealment',
    label: 'Someone may have hidden an item',
    operatorHint:
      'A hand moved to the body in a way that can mean hiding something. Adjusting clothes or pocketing a phone looks the same.',
    whatToDo: 'Watch the clip before acting. This one is wrong often — do not approach on the alert alone.',
    component: 'component_3',
    glyph: '◫',
  },
  rapid_approach: {
    type: 'rapid_approach',
    label: 'Someone moved quickly at another person',
    operatorHint:
      'A person moved fast towards someone else. Running for a lift or greeting a friend looks the same.',
    whatToDo: 'Watch the clip. Check whether the other person backed away.',
    component: 'component_3',
    glyph: '➤',
  },

  // ---- C4 — Fall & Abandoned-Object Detection ----
  fall: {
    type: 'fall',
    label: 'Someone may have fallen',
    operatorHint:
      'A person went down quickly and stayed down. Sitting on the floor, bending, kneeling or picking something up can all look like this — the thing to check is whether they got back up.',
    whatToDo: 'Watch the clip now. If the person is still down, send someone to check on them.',
    component: 'component_4',
    glyph: '↯',
  },
  abandoned_object: {
    type: 'abandoned_object',
    label: 'Item left on its own',
    operatorHint:
      'Something was put down and the person who seemed to leave it walked away. The system is guessing who the owner is from who was standing nearest — it can easily be wrong.',
    whatToDo: 'Check whether the owner is still in view nearby. If not, follow your unattended-item procedure.',
    component: 'component_4',
    glyph: '▣',
  },
};

/**
 * Turn any incident_type — known or not — into something displayable.
 * An unknown type from a teammate's new detector renders as a readable label
 * with an "unregistered type" note, instead of breaking the queue.
 */
export function describeIncident(type: string, componentId: ComponentId): IncidentMeta & { known: boolean } {
  const known = INCIDENT_TYPES[type];
  if (known) return { ...known, known: true };

  return {
    type,
    label: humanise(type),
    operatorHint:
      'This is a new kind of alert the dashboard has not been told about yet. What the camera saw is shown below as the system reported it.',
    whatToDo: 'Watch the clip and use your judgement. Tell the project team this alert type needs a proper description.',
    component: componentId,
    glyph: '◆',
    known: false,
  };
}

/** "restricted_zone_intrusion" -> "Restricted zone intrusion" */
export function humanise(snake: string): string {
  if (!snake) return 'Unknown';
  const spaced = snake.replace(/[_-]+/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** All registered incident types, grouped for the filter panel. */
export function incidentTypesByComponent(): Record<ComponentId, IncidentMeta[]> {
  const grouped = {
    component_1: [],
    component_2: [],
    component_3: [],
    component_4: [],
  } as Record<ComponentId, IncidentMeta[]>;

  for (const meta of Object.values(INCIDENT_TYPES)) {
    grouped[meta.component].push(meta);
  }
  return grouped;
}
