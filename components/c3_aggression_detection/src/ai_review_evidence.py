"""Append an explicitly provisional AI visual review without granting supervision."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from src.foundation_io import sha256, write_csv, write_json
from src.human_review import PACK_SCHEMA, validate_record
from src.lock_splits import load_locked
from src.reconcile_reviews import coverage_report, normalize


AI_SCHEMA = 'c3_ai_visual_review_v1'
AI_STATUS = 'provisional_ai_review'


def normalize_ai_reviews(evidence, pack, pack_hash, rows, location):
    if evidence.get('schema') != AI_SCHEMA or evidence.get('pack_sha256') != pack_hash:
        raise ValueError('AI review must use its own schema and matching review pack')
    if evidence.get('reviewer_type') != 'ai' or evidence.get('independent_human_review') is not False:
        raise ValueError('AI observations cannot claim independent human review')
    if not evidence.get('review_method') or not evidence.get('sampling_limitations'):
        raise ValueError('AI review requires an explicit observation method and limitations')
    if pack.get('schema') != PACK_SCHEMA:
        raise ValueError('Invalid source review pack')
    clips = {c['clip_id']: c for c in pack['clips']}
    by_id = {r['video_id']: r for r in rows}
    records = []
    seen = set()
    for i, raw in enumerate(evidence['records'], 1):
        key = raw.get('clip_id')
        if key not in clips or key in seen:
            raise ValueError('Unknown or duplicate clip in AI observations')
        seen.add(key)
        clip = clips[key]
        row = by_id.get(clip['video_id'])
        if row is None or row['split'] not in ('train', 'val') or clip['split'] != row['split']:
            raise ValueError('Test/unknown/mismatched source in development AI review')
        if clip['source_sha256'] != row['content_sha256'] or clip['leakage_group'] != row['leakage_group']:
            raise ValueError('AI source identity does not match locked manifest')
        if raw.get('review_status') in ('independently_verified', 'verified') or raw.get('independent_human_review'):
            raise ValueError('AI decisions must remain provisional')
        # AI files use proposal/observation field names, not human-verification fields.
        if any(k in raw for k in ('verified_fight_intervals', 'first_physical_contact_s')):
            raise ValueError('Use AI proposal/observation fields instead of human-verification fields')
        canonical_input = dict(raw, precursors=raw.get('proposed_precursors', []),
            first_physical_contact_s=raw.get('proposed_contact_s'),
            verified_fight_intervals=raw.get('observed_fight_intervals', []))
        canonical = validate_record(canonical_input, clip, float(row['duration_s']))
        if not str(raw.get('review_notes', '')).strip():
            raise ValueError('Every AI decision needs visible-evidence/uncertainty notes')
        samples = raw.get('observed_frame_times_s', [])
        if not samples or any(not isinstance(t, (float, int)) or not math.isfinite(t) or
                not clip['clip_start_s'] <= t <= clip['clip_end_s'] for t in samples):
            raise ValueError('AI observations need sampled original-video timestamps within the clip')
        bracket = raw.get('contact_uncertainty_s')
        if bracket is not None:
            if not isinstance(bracket, list) or len(bracket) != 2:
                raise ValueError('Contact uncertainty needs [earliest,latest] bounds')
            a, b = map(float, bracket)
            contact = canonical['first_physical_contact_s']
            if not (math.isfinite(a) and math.isfinite(b) and
                    clip['clip_start_s'] <= a < b <= clip['clip_end_s'] and
                    contact is not None and a <= contact <= b):
                raise ValueError('Contact estimate must lie inside its uncertainty bracket')
            if any(end > a for start, end in canonical['precursors']):
                raise ValueError('Proposed precursor must precede the earliest plausible contact')
        elif canonical['review_label'] == 'precursor':
            raise ValueError('AI precursor candidates require explicit contact uncertainty')
        original = dict(raw, **canonical, reviewer='Codex_AI_visual_review',
                        reviewer_type='ai', independent_human_review=False)
        record = normalize(original, row, f'{location}:{i}', 'manual_json')
        if record['issues']:
            raise ValueError(f'Invalid AI observation {key}: {record["issues"]}')
        record['original'] = dict(raw, reviewer_type='ai', independent_human_review=False)
        record.update(review_status=AI_STATUS, reviewer_type='ai',
            eligible_for_fitting=False, requires_human_adjudication=True,
            supervision_policy='No verified targets; no implicit AI/weak-label training',
            submitted=dict(raw))
        records.append(record)
    if seen != set(clips):
        raise ValueError('This pilot review must account for every prepared clip')
    return records


def run(args):
    rows = load_locked(args.locked)
    expected = sha256(args.locked)
    pack = json.loads(Path(args.pack).read_text(encoding='utf-8'))
    base = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    evidence = json.loads(Path(args.reviews).read_text(encoding='utf-8'))
    if base.get('schema') != 'c3_review_evidence_v1' or any(x.get('locked_manifest_sha256') != expected for x in (pack, base)):
        raise ValueError('Base annotations/review pack belong to different locked splits')
    if pack.get('base_annotations_sha256') != sha256(args.annotations):
        raise ValueError('Review pack belongs to different parent evidence')
    pack_root = Path(args.pack).resolve().parent
    for clip in pack['clips']:
        media = (pack_root/clip['media']).resolve()
        if not media.is_relative_to(pack_root) or sha256(media) != clip['media_sha256']:
            raise ValueError('Changed/invalid review media')
    location = str(Path(args.reviews).resolve())
    added = normalize_ai_reviews(evidence, pack, sha256(args.pack), rows, location)
    records = list(base['records']) + added
    out = Path(args.out_dir)
    doc = dict(base, records=records, parent_annotations_sha256=sha256(args.annotations),
        ai_visual_review_sha256=sha256(args.reviews),
        source_sha256=dict(base.get('source_sha256', {}), **{location: sha256(args.reviews)}))
    write_json(out/'annotations.json', doc)
    report = coverage_report(records)
    report.update(ai_review_records=len(added),
        ai_proposed_labels=dict(Counter(r['label'] for r in added)),
        ai_labels_by_split={s: dict(Counter(r['label'] for r in added if r['split'] == s)) for s in ('train', 'val')},
        verified_labels_created=0, model_trained=False,
        note='AI frame-sampled observations are provisional; all are masked by the verified-supervision loader.')
    write_json(out/'annotation_coverage.json', report)
    write_csv(out/'ai_review_summary.csv', [dict(clip_id=raw['clip_id'], video_id=r['video_id'],
        split=r['split'], review_label=r['label'], review_status=AI_STATUS,
        reviewed_ranges_s=json.dumps(r['coverage_s']), precursors=json.dumps(r['precursors']),
        proposed_contact_s=r['contact_s'], contact_uncertainty_s=json.dumps(raw.get('contact_uncertainty_s')),
        observed_fight_intervals=json.dumps(r['fight_intervals']),
        eligible_for_verified_fitting=False, review_notes=raw['review_notes'])
        for raw, r in zip(evidence['records'], added)])
    print(json.dumps({k: report[k] for k in ('ai_review_records', 'ai_proposed_labels',
        'ai_labels_by_split', 'verified_labels_created', 'model_trained')}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--annotations', default='data/foundation/v1/annotations.json')
    p.add_argument('--pack', default='data/reviewed_annotations/quick_human_v1/pack.json')
    p.add_argument('--reviews', default='data/reviewed_annotations/ai_visual_v1/labels.json')
    p.add_argument('--out-dir', default='data/foundation/ai_visual_v1')
    run(p.parse_args())


if __name__ == '__main__':
    main()
