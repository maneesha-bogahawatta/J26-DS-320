"""Strict, append-only human review of visible cue/contact/ongoing violence.

Reuses the existing immutable human-review media pack and normalized training
schema. Weak annotated onset never supplies contact in this workflow. No model
is trained, no test media is reviewed, and no existing decision is replaced.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
import os
from pathlib import Path

from src.foundation_io import sha256, write_json, write_new
from src.human_review import PACK_SCHEMA, LABELS
from src.lock_splits import load_locked
from src.reconcile_reviews import coverage_report, intervals, normalize


FORM_SCHEMA = 'c3_visible_contact_human_review_v1'
CONTACT_DEFINITION = 'visible_strike_push_kick_or_grapple_v1'
CONTACT_STATES = ('not_observed', 'uncertain', 'confirmed_visible', 'already_in_progress')


def _ranges(value, duration, name):
    result = sorted(intervals(value, duration))
    if any(b > x for (_, b), (x, _) in zip(result, result[1:])):
        raise ValueError(f'{name} intervals must not overlap')
    return result


def _point(value, coverage, name):
    if value in ('', None):
        return None
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a numeric original-video time')
    point = float(value)
    if not math.isfinite(point) or not any(a <= point <= b for a, b in coverage):
        raise ValueError(f'{name} must lie inside reviewed coverage')
    return point


def validate_contact_record(raw, clip, duration):
    """Canonical semantic evidence for exact two-human agreement.

    A confirmed timestamp is a human observation of the first contact under the
    versioned definition, not an imported dataset boundary. An uncertainty
    bracket is retained, but cannot establish a verified precursor target.
    """
    label = raw.get('review_label', 'unreviewed')
    if label not in LABELS:
        raise ValueError('Unknown review label')
    if raw.get('timestamp_basis') != 'original_video_seconds':
        raise ValueError('All review times must use original_video_seconds')
    if raw.get('contact_definition') != CONTACT_DEFINITION:
        raise ValueError('The visible physical-contact definition must be explicit')
    coverage = _ranges(raw.get('reviewed_ranges_s'), duration, 'Reviewed coverage')
    cues = _ranges(raw.get('precursors'), duration, 'Cue')
    fight = _ranges(raw.get('verified_fight_intervals'), duration, 'Fight')
    if label != 'unreviewed' and not coverage:
        raise ValueError('Completed decisions require reviewed coverage')
    if any(not clip['clip_start_s'] <= a < b <= clip['clip_end_s'] for a, b in coverage):
        raise ValueError('Reviewed coverage must stay inside the media clip')
    if any(not any(x <= a < b <= y for x, y in coverage) for a, b in cues + fight):
        raise ValueError('Cue/fight intervals must stay inside reviewed coverage')
    first_cue = _point(raw.get('first_threatening_cue_s'), coverage, 'First threatening cue')
    contact = _point(raw.get('first_physical_contact_s'), coverage, 'First physical contact')
    state = raw.get('contact_observation', 'not_observed')
    if state not in CONTACT_STATES:
        raise ValueError('Unknown contact observation state')
    bracket = raw.get('contact_uncertainty_s')
    if bracket in ('', None, []):
        bracket = None
    else:
        if not isinstance(bracket, (list, tuple)) or len(bracket) != 2:
            raise ValueError('Contact uncertainty requires [earliest,latest] seconds')
        lo = _point(bracket[0], coverage, 'Earliest plausible contact')
        hi = _point(bracket[1], coverage, 'Latest plausible contact')
        if lo is None or hi is None or lo > hi or not any(a <= lo <= hi <= b for a, b in coverage):
            raise ValueError('Invalid contact uncertainty bracket')
        bracket = [lo, hi]
    visible_attestation = raw.get('contact_personally_observed') is True
    not_from_annotation = raw.get('contact_not_inferred_from_dataset_onset') is True
    if state == 'confirmed_visible':
        if contact is None or not visible_attestation or not not_from_annotation:
            raise ValueError('Confirmed contact needs personal visible observation, not a dataset onset')
        if bracket is not None:
            raise ValueError('Uncertain contact cannot be confirmed; retain an uncertain decision')
    elif contact is not None:
        raise ValueError('Only confirmed_visible may assert a first physical-contact timestamp')
    if state == 'uncertain' and label not in ('uncertain', 'unreviewed', 'active_fight', 'already_violent'):
        raise ValueError('Uncertain contact cannot establish a verified precursor or negative')
    if bracket is not None and state != 'uncertain':
        raise ValueError('Contact uncertainty bracket requires uncertain contact observation')
    if label == 'precursor':
        if not cues or first_cue is None or state != 'confirmed_visible':
            raise ValueError('Precursor requires first threatening cue, cue intervals and confirmed visible contact')
        if first_cue != cues[0][0] or any(b >= contact for a, b in cues):
            raise ValueError('Observed cue intervals must strictly precede contact; first cue is their first start')
        if not any(max(a, contact - 3.) < b for a, b in cues):
            raise ValueError('A precursor target needs observed cue coverage within three seconds before contact')
    if contact is not None and any(a < contact for a, b in fight):
        raise ValueError('Ongoing physical fight cannot precede confirmed first contact')
    if label == 'non_aggressive' and (cues or fight or first_cue is not None or contact is not None or state != 'not_observed'):
        raise ValueError('Non-aggressive coverage cannot assert cues, contact, uncertainty or violence')
    if label in ('active_fight', 'already_violent') and not fight:
        raise ValueError('Ongoing violence needs explicit fight intervals')
    if label == 'already_violent' and state not in ('already_in_progress', 'uncertain'):
        raise ValueError('Already violent clips cannot invent their unobserved first contact')
    if label not in ('precursor', 'uncertain', 'unreviewed') and (cues or first_cue is not None):
        raise ValueError('Threatening cue markers require precursor or uncertain evidence')
    return dict(review_label=label, timestamp_basis='original_video_seconds',
        reviewed_ranges_s=coverage, precursors=cues, first_threatening_cue_s=first_cue,
        first_physical_contact_s=contact, verified_fight_intervals=fight,
        contact_observation=state, contact_uncertainty_s=bracket,
        contact_definition=CONTACT_DEFINITION, contact_personally_observed=visible_attestation,
        contact_not_inferred_from_dataset_onset=not_from_annotation)


def blank_contact_form(pack, pack_hash, split):
    if split not in ('train', 'val'):
        raise ValueError('Development reviews exclude the locked test set')
    return dict(schema=FORM_SCHEMA, pack_sha256=pack_hash, split=split,
        reviewer='', independent_human_review=False, records=[dict(
            clip_id=c['clip_id'], review_label='unreviewed', timestamp_basis='original_video_seconds',
            reviewed_ranges_s=[[c['clip_start_s'], c['clip_end_s']]], precursors=[],
            first_threatening_cue_s=None, first_physical_contact_s=None,
            verified_fight_intervals=[], contact_observation='not_observed',
            contact_uncertainty_s=None, contact_definition=CONTACT_DEFINITION,
            contact_personally_observed=False, contact_not_inferred_from_dataset_onset=False,
            review_notes='') for c in pack['clips'] if c['split'] == split])


def _validate_pack(pack, rows):
    if pack.get('schema') != PACK_SCHEMA:
        raise ValueError('Expected the existing immutable human-review pack')
    by_id = {r['video_id']: r for r in rows}
    clips = {c['clip_id']: c for c in pack['clips']}
    if len(clips) != len(pack['clips']):
        raise ValueError('Duplicate review clip identities')
    for clip in clips.values():
        row = by_id.get(clip['video_id'])
        if row is None or row['split'] not in ('train', 'val') or clip['split'] != row['split']:
            raise ValueError('Unknown, held-out test, or mismatched development split')
        if clip['source_sha256'] != row['content_sha256'] or clip['leakage_group'] != row['leakage_group']:
            raise ValueError('Review source differs from locked identity')
    return by_id, clips


def import_contact_evidence(pack, pack_hash, rows, base, forms):
    """Append raw submissions and exact semantic consensus; never edit parents."""
    by_id, clips = _validate_pack(pack, rows)
    votes = defaultdict(list)
    added = []
    seen = set()
    for location, form in forms:
        if form.get('schema') != FORM_SCHEMA or form.get('pack_sha256') != pack_hash:
            raise ValueError('Contact review belongs to a different schema or media pack')
        if form.get('split') not in ('train', 'val') or not isinstance(form.get('records'), list):
            raise ValueError('Contact forms require a train/val split and records')
        reviewer = str(form.get('reviewer') or '').strip()
        local_seen = set()
        for index, raw in enumerate(form['records']):
            clip_id = raw.get('clip_id')
            if clip_id not in clips or clip_id in local_seen:
                raise ValueError('Unknown or duplicate clip in contact review')
            local_seen.add(clip_id)
            clip = clips[clip_id]
            row = by_id[clip['video_id']]
            if clip['split'] != form['split']:
                raise ValueError('Train and validation review decisions remain separate')
            canonical = validate_contact_record(raw, clip, float(row['duration_s']))
            if canonical['review_label'] != 'unreviewed':
                if not reviewer or form.get('independent_human_review') is not True:
                    raise ValueError('Completed contact decisions require a real reviewer ID and human attestation')
                key = (clip_id, reviewer.casefold())
                if key in seen:
                    raise ValueError('A reviewer cannot provide a second independent vote')
                seen.add(key)
            original = dict(raw, **canonical, reviewer=reviewer,
                independent_human_review=form.get('independent_human_review', False),
                source_form=str(location), source_schema=FORM_SCHEMA)
            record = normalize(original, row, f'{location}:{index+1}', 'manual_json')
            record.update(source_schema=FORM_SCHEMA, submitted=copy.deepcopy(raw),
                contact_semantics=copy.deepcopy(canonical))
            added.append(record)
            if canonical['review_label'] != 'unreviewed':
                votes[clip_id].append((canonical, reviewer, record['record_id']))
    agreed, disagreements = [], []
    for key, decisions in sorted(votes.items()):
        if len(decisions) > 2:
            raise ValueError('At most two reviewers per clip in one import; adjudicate in a new version')
        if len(decisions) < 2:
            continue
        first, second = decisions
        if first[0] != second[0]:
            disagreements.append(key)
            continue
        if first[0]['review_label'] in ('uncertain', 'unreviewed'):
            continue
        original = dict(first[0], reviewer_1=first[1], reviewer_2=second[1],
            independent_human_review=True, source_schema=FORM_SCHEMA,
            evidence_record_ids=[first[2], second[2]],
            agreement_policy='Exact two-human cue/contact/fight semantic agreement; notes retained separately')
        row = by_id[clips[key]['video_id']]
        record = normalize(original, row, f'visible_contact_agreement:{pack_hash}:{key}', 'manual_json')
        if record['review_status'] != 'independently_verified' or record['issues']:
            raise ValueError('Agreed visible-contact evidence failed normalization')
        record.update(source_schema=FORM_SCHEMA, contact_semantics=copy.deepcopy(first[0]))
        added.append(record)
        agreed.append(key)
    records = copy.deepcopy(base['records']) + added
    report = coverage_report(records)
    report.update(added_records=len(added), agreed_clips=agreed, disagreement_clips=disagreements,
        single_review_clips=[k for k, v in votes.items() if len(v) == 1],
        unreviewed_clips=[k for k in clips if k not in votes],
        strict_contact_readiness=readiness_report(records), trained_model=False,
        verification_limit='Reviewer IDs and attestations cannot authenticate independent human work.')
    return records, report


def readiness_report(records):
    """Annotation-only minimum guard, never a scientific-accuracy assertion.

    Test records are excluded. Cache quality and eligible-window readiness must
    still be checked by train_stgcn before any fit. Legacy verified records stay
    preserved but do not count toward this stricter contact-semantics protocol.
    """
    counts = {s: {'precursor_positive_videos': 0, 'non_aggressive_videos': 0} for s in ('train', 'val')}
    source_groups = {s: {'precursor': set(), 'non_aggressive': set()} for s in counts}
    legacy = 0
    rejected = 0
    for record in records:
        if record.get('split') not in counts or record.get('review_status') != 'independently_verified' or record.get('issues'):
            continue
        if record.get('source_schema') != FORM_SCHEMA:
            legacy += 1
            continue
        try:
            semantic = record['contact_semantics']
            coverage = record['coverage_s']
            canonical = validate_contact_record(semantic,
                dict(clip_start_s=min(a for a, b in coverage), clip_end_s=max(b for a, b in coverage)),
                max(b for a, b in coverage))
            if canonical['review_label'] != record['label'] or canonical['precursors'] != record['precursors'] or canonical['first_physical_contact_s'] != record['contact_s']:
                raise ValueError('Normalized contact evidence disagrees with its semantic source')
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        if record.get('label') in source_groups[record['split']]:
            source_groups[record['split']][record['label']].add(record['video_id'])
    for split, groups in source_groups.items():
        counts[split] = dict(precursor_positive_videos=len(groups['precursor']),
            non_aggressive_videos=len(groups['non_aggressive']))
    blockers = [f'{split}: no strict-contact independently verified {kind.replace("_", " ")}'
        for split, values in counts.items() for kind, count in values.items() if count == 0]
    return dict(status='blocked' if blockers else 'annotation_minimum_met_cache_checks_required',
        by_split=counts, blockers=blockers, excluded_legacy_verified_records=legacy,
        rejected_malformed_strict_records=rejected,
        verified_early_warning_accuracy=None, verified_early_warning_recall=None,
        scientific_validation_achieved=False, model_trained=False,
        notes=['Locked test records do not contribute to development readiness.',
            'Minimum label presence does not establish event diversity, calibration, or deployment readiness.',
            'Run the existing train_stgcn --check-only against compatible caches before fitting.'])


def _load(args):
    rows = load_locked(args.locked)
    pack_path = Path(args.pack).resolve()
    pack = json.loads(pack_path.read_text(encoding='utf-8'))
    _validate_pack(pack, rows)
    if pack.get('locked_manifest_sha256') != sha256(args.locked):
        raise ValueError('Media pack differs from locked split manifest')
    for clip in pack['clips']:
        media = (pack_path.parent/clip['media']).resolve()
        if not media.is_relative_to(pack_path.parent) or sha256(media) != clip['media_sha256']:
            raise ValueError('Review media changed or escaped immutable pack')
    return rows, pack, pack_path


def prepare(args):
    rows, pack, pack_path = _load(args)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('contact_review_ui.html').read_text(encoding='utf-8')
    for split in ('train', 'val'):
        form = blank_contact_form(pack, sha256(pack_path), split)
        clips = [dict(c, media=Path(os.path.relpath(pack_path.parent/c['media'], output/split)).as_posix())
            for c in pack['clips'] if c['split'] == split]
        encoded = json.dumps(dict(form=form, clips=clips), ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
        write_new(output/split/'review.html', template.replace('__REVIEW_CONFIG__', encoded))
        write_json(output/split/'review.template.json', form)
    summary = dict(schema=FORM_SCHEMA, media_pack_sha256=sha256(pack_path),
        clips_by_split=dict(Counter(c['split'] for c in pack['clips'])),
        new_media_extraction=False, verified_labels_created=0, model_trained=False)
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def import_forms(args):
    rows, pack, pack_path = _load(args)
    base = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    if base.get('schema') != 'c3_review_evidence_v1' or base.get('locked_manifest_sha256') != sha256(args.locked):
        raise ValueError('Parent annotations differ from locked splits')
    forms = [(str(Path(p).resolve()), json.loads(Path(p).read_text(encoding='utf-8-sig'))) for p in args.reviews]
    records, report = import_contact_evidence(pack, sha256(pack_path), rows, base, forms)
    document = dict(base, records=records, parent_annotations_sha256=sha256(args.annotations),
        strict_contact_pack_sha256=sha256(pack_path), contact_review_schema=FORM_SCHEMA,
        source_sha256=dict(base.get('source_sha256', {}), **{str(Path(p).resolve()): sha256(p) for p in args.reviews}))
    # Match existing versioned writers: identical reruns allowed, changed content refused.
    write_json(Path(args.out_dir)/'annotations.json', document)
    write_json(Path(args.out_dir)/'annotation_coverage.json', report)
    print(json.dumps(report['strict_contact_readiness'], indent=2))
    return report


def audit(args):
    base = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    rows = load_locked(args.locked)
    if base.get('schema') != 'c3_review_evidence_v1' or base.get('locked_manifest_sha256') != sha256(args.locked):
        raise ValueError('Annotation audit must match locked split provenance')
    by_id = {r['video_id']: r for r in rows}
    if any(r['video_id'] not in by_id or r['split'] != by_id[r['video_id']]['split'] for r in base['records']):
        raise ValueError('Annotation identities/splits differ from locked manifest')
    report = dict(schema='c3_contact_semantics_audit_v1', annotations_sha256=sha256(args.annotations),
        records=len(base['records']), status_counts=dict(Counter(r['review_status'] for r in base['records'])),
        strict_contact_readiness=readiness_report(base['records']),
        dataset_onset_equals_visible_first_contact_assumed=False,
        inspected_test_media=False, changed_existing_labels=False,
        schema_findings=[
            'UBI binary intervals supervise weak annotation onset and active fight, not visible first contact.',
            'Existing human_review_v1 preserves independent review but lacks explicit contact-definition/observation fields.',
            'Historical approved-manifest normalization can derive contact from approved interval onset; retained as legacy evidence.',
            'AI/v3/v4 proposals remain provisional and cannot establish independently verified pre-contact aggression.',
            'New strict contact forms separately mark first threatening cue, visible physical contact and ongoing fight.'])
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'import', 'audit'):
        item = commands.add_parser(name)
        item.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
        if name != 'audit':
            item.add_argument('--pack', default='data/reviewed_annotations/quick_human_v1/pack.json')
            item.add_argument('--out-dir', required=True)
        if name != 'prepare':
            item.add_argument('--annotations', default='data/foundation/ai_visual_v1/annotations.json')
        if name == 'import':
            item.add_argument('--reviews', nargs='+', required=True)
        if name == 'audit':
            item.add_argument('--out', required=True)
    args = parser.parse_args()
    return {'prepare': prepare, 'import': import_forms, 'audit': audit}[args.command](args)


if __name__ == '__main__':
    main()
