"""Prepare blind cached-clip reviews and append independently agreed evidence.

No inference, automatic labels, original-review edits, or model fitting occurs.
"""
import argparse
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from src.foundation_io import sha256, write_csv, write_json, write_new
from src.lock_splits import load_locked
from src.pose_sequences import CACHE_SCHEMA, load_cache
from src.reconcile_reviews import coverage_report, intervals, normalize


PACK_SCHEMA = 'c3_human_review_pack_v1'
FORM_SCHEMA = 'c3_independent_human_review_v1'
LABELS = ('unreviewed', 'precursor', 'non_aggressive', 'active_fight',
          'already_violent', 'uncertain')


def plan_clips(rows, spans, seconds=8.):
    """Use cached context only; weak onsets select context, never a target."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Clip duration must be positive and finite')
    by_id = {r['video_id']: r for r in rows}
    result = []
    groups = set()
    for key, (lo, hi) in sorted(spans.items()):
        row = by_id[key]
        if row['split'] not in ('train', 'val'):
            raise ValueError('Test videos are prohibited in the development review pack')
        if row['leakage_group'] in groups:
            raise ValueError('Review pack must use distinct locked source groups')
        groups.add(row['leakage_group'])
        if not (math.isfinite(lo) and math.isfinite(hi) and
                0 <= lo < hi <= float(row['duration_s']) + .001):
            raise ValueError('Invalid cached context bounds')
        onsets = [a for a, b in intervals(row.get('intervals'), float(row['duration_s']))
                  if lo <= a < hi]
        start = max(lo, min(onsets) - 5.) if onsets else max(lo, (lo + hi - seconds) / 2.)
        end = min(hi, start + seconds)
        result.append(dict(clip_id=f'{row["split"]}_{len(result)+1:03d}',
            video_id=key, split=row['split'], leakage_group=row['leakage_group'],
            source_video=row['video'], source_sha256=row['content_sha256'],
            clip_start_s=round(start, 6), clip_end_s=round(end, 6),
            timestamp_basis='original_video_seconds'))
    return result


def cached_spans(locked, indexes):
    """Authenticate cache identities before choosing review footage."""
    rows = load_locked(locked)
    by_id = {r['video_id']: r for r in rows}
    spans = {}
    for split, path in indexes.items():
        index = json.loads(Path(path).read_text(encoding='utf-8'))
        if index.get('schema') != CACHE_SCHEMA or index.get('locked_manifest_sha256') != sha256(locked):
            raise ValueError('Cache index does not match the locked split manifest')
        for key, entry in index['entries'].items():
            if key not in by_id or by_id[key]['split'] != split or split not in ('train', 'val'):
                raise ValueError('Review index has an unknown or held-out/wrong-split source')
            if key in spans or sha256(entry['path']) != entry['cache_sha256']:
                raise ValueError('Duplicate source or changed cache')
            arrays, meta = load_cache(entry['path'])
            if meta['video_id'] != key or meta['source_sha256'] != by_id[key]['content_sha256']:
                raise ValueError('Cache/source identity mismatch')
            lo = float(meta['start_s'])
            hi = min(float(by_id[key]['duration_s']),
                     float(meta.get('requested_end_s') or meta['source_duration_s']))
            # Refuse clips extending beyond actually extracted context by more than one sample.
            if lo < float(arrays['t'][0]) - 1./meta['sample_fps'] - .001 or hi > float(arrays['t'][-1]) + 1./meta['sample_fps'] + .001:
                raise ValueError('Requested bounds exceed complete cached context')
            spans[key] = (lo, hi)
    return rows, spans


def export_clip(clip, dest, ffmpeg):
    """Browser-compatible silent review copy, without annotation/heuristic overlays."""
    if sha256(clip['source_video']) != clip['source_sha256']:
        raise ValueError('Source video changed after locking')
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite review media: {dest}')
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-ss', str(clip['clip_start_s']), '-i', clip['source_video'],
        '-t', str(clip['clip_end_s'] - clip['clip_start_s']), '-map', '0:v:0', '-an',
        '-vf', "scale=w='min(960,iw)':h=-2", '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)],
        check=True, timeout=60, capture_output=True, text=True)
    return sha256(dest)


def blank_form(pack, digest, split):
    return dict(schema=FORM_SCHEMA, pack_sha256=digest, split=split, reviewer='',
        independent_human_review=False, records=[dict(clip_id=c['clip_id'],
            review_label='unreviewed', reviewed_ranges_s=[[c['clip_start_s'], c['clip_end_s']]],
            precursors=[], first_physical_contact_s=None, verified_fight_intervals=[],
            review_notes='') for c in pack['clips'] if c['split'] == split])


def prepare(args):
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise FileNotFoundError('Local FFmpeg is required to export review copies; no download attempted')
    rows, spans = cached_spans(args.locked, {'train': args.train_index, 'val': args.val_index})
    base = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    if base.get('schema') != 'c3_review_evidence_v1' or base['locked_manifest_sha256'] != sha256(args.locked):
        raise ValueError('Base annotations do not match locked splits')
    clips = plan_clips(rows, spans)
    if not clips:
        raise ValueError('No complete cached context to review')
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for i, clip in enumerate(clips, 1):
        clip['media'] = f'{clip["split"]}/clips/{clip["clip_id"]}.mp4'
        clip['media_sha256'] = export_clip(clip, out/clip['media'], ffmpeg)
        print(f'Review copy {i}/{len(clips)}: {clip["clip_id"]}', flush=True)
    pack = dict(schema=PACK_SCHEMA, locked_manifest_sha256=sha256(args.locked),
        base_annotations_sha256=sha256(args.annotations),
        cache_index_sha256={s: sha256(p) for s, p in
            (('train', args.train_index), ('val', args.val_index))}, clips=clips,
        selection='Retrospective diagnostic pilot from cached contexts; not a benchmark',
        supervision='All forms start unreviewed; no heuristic proposals are review decisions')
    write_json(out/'pack.json', pack)
    digest = sha256(out/'pack.json')
    template = Path(__file__).with_name('human_review_ui.html').read_text(encoding='utf-8')
    for split in ('train', 'val'):
        form = blank_form(pack, digest, split)
        config = dict(form=form, clips=[dict(c, media='clips/'+Path(c['media']).name)
                      for c in clips if c['split'] == split])
        # Prevent a source path or note from ending a script tag in the standalone UI.
        encoded = json.dumps(config, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
        write_new(out/split/'review.html', template.replace('__REVIEW_CONFIG__', encoded))
        write_json(out/split/'review.template.json', form)
        write_csv(out/split/'queue.csv', [c for c in clips if c['split'] == split])
    summary = dict(clips=len(clips), by_split=dict(Counter(c['split'] for c in clips)),
        source_seconds=round(sum(c['clip_end_s']-c['clip_start_s'] for c in clips), 6),
        pose_extraction_performed=False, verified_labels_created=0, trained_model=False,
        pack_sha256=digest)
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return pack


def validate_record(raw, clip, duration):
    """Reject malformed/out-of-clip evidence before normalizing human decisions."""
    label = raw.get('review_label', 'unreviewed')
    if label not in LABELS:
        raise ValueError('Unknown human review label')
    coverage = intervals(raw.get('reviewed_ranges_s'), duration)
    cues = intervals(raw.get('precursors'), duration)
    fight = intervals(raw.get('verified_fight_intervals'), duration)
    for a, b in coverage:
        if not clip['clip_start_s'] <= a < b <= clip['clip_end_s']:
            raise ValueError('Reviewed coverage must stay inside this review clip')
    if label != 'unreviewed' and not coverage:
        raise ValueError('A human decision requires explicit reviewed coverage')
    if any(not any(x <= a < b <= y for x, y in coverage) for a, b in cues+fight):
        raise ValueError('Cue/fight intervals must stay inside reviewed coverage')
    contact = raw.get('first_physical_contact_s')
    contact = None if contact in ('', None) else float(contact)
    if contact is not None and not (math.isfinite(contact) and clip['clip_start_s'] <= contact <= clip['clip_end_s']):
        raise ValueError('Confirmed contact must be visible inside this review clip')
    if label == 'precursor' and (not cues or contact is None or any(b > contact for a, b in cues)):
        raise ValueError('Precursor needs observed cues preceding a confirmed contact')
    if label == 'non_aggressive' and (cues or fight or contact is not None):
        raise ValueError('Non-aggressive coverage cannot also assert cues/contact/fight')
    if label in ('active_fight', 'already_violent') and not fight:
        raise ValueError('Active violence requires explicit reviewed fight intervals')
    if label not in ('precursor', 'uncertain', 'unreviewed') and cues:
        raise ValueError('Cue intervals require a precursor or uncertain decision')
    return dict(review_label=label, reviewed_ranges_s=sorted(coverage),
        precursors=sorted(cues), first_physical_contact_s=contact,
        verified_fight_intervals=sorted(fight))


def import_evidence(pack, pack_hash, rows, base, forms):
    """Append every raw decision; verify only matching, attested independent reviews.

    Reviewer IDs and attestations record provenance; software cannot authenticate
    independent human work. Disagreements remain provisional, without averaging.
    """
    if pack.get('schema') != PACK_SCHEMA:
        raise ValueError('Invalid review pack schema')
    by_id = {r['video_id']: r for r in rows}
    clips = {c['clip_id']: c for c in pack['clips']}
    if len(clips) != len(pack['clips']):
        raise ValueError('Duplicate clip IDs in review pack')
    for clip in clips.values():
        row = by_id.get(clip['video_id'])
        if row is None or row['split'] not in ('train', 'val') or clip['split'] != row['split']:
            raise ValueError('Test/unknown/mismatched split in human review pack')
        if clip['source_sha256'] != row['content_sha256'] or clip['leakage_group'] != row['leakage_group']:
            raise ValueError('Review source identity differs from locked manifest')
    added = []
    votes = defaultdict(list)
    seen = set()
    for location, form in forms:
        if form.get('schema') != FORM_SCHEMA or form.get('pack_sha256') != pack_hash:
            raise ValueError('Review form does not match the immutable pack')
        reviewer = str(form.get('reviewer') or '').strip()
        entries = form.get('records')
        if not isinstance(entries, list) or form.get('split') not in ('train', 'val'):
            raise ValueError('Review form needs records and a train/val role')
        local_seen = set()
        for i, raw in enumerate(entries):
            key = raw.get('clip_id')
            if key not in clips or key in local_seen:
                raise ValueError('Unknown/duplicate clip ID in review form')
            local_seen.add(key)
            clip = clips[key]
            if clip['split'] != form['split']:
                raise ValueError('Train and validation review forms must remain separate')
            canonical = validate_record(raw, clip, float(by_id[clip['video_id']]['duration_s']))
            if canonical['review_label'] != 'unreviewed':
                if not reviewer or form.get('independent_human_review') is not True:
                    raise ValueError('Completed decisions need a reviewer ID and independent-human attestation')
                identity = (key, reviewer.casefold())
                if identity in seen:
                    raise ValueError('The same reviewer cannot supply both independent decisions')
                seen.add(identity)
            original = dict(raw, **canonical, reviewer=reviewer,
                independent_human_review=form.get('independent_human_review', False),
                source_form=str(location), source_video=clip['source_video'])
            record = normalize(original, by_id[clip['video_id']], f'{location}:{i+1}', 'manual_json')
            # Keep the exact submitted fields/notes as well as normalized evidence.
            record['submitted'] = dict(raw)
            added.append(record)
            if canonical['review_label'] != 'unreviewed':
                votes[key].append((canonical, reviewer, record['record_id']))
    agreed = []
    disagreements = []
    for key, decisions in sorted(votes.items()):
        if len(decisions) > 2:
            raise ValueError('At most two independent decisions per clip per import; adjudicate in a new version')
        if len(decisions) != 2:
            continue
        first, second = decisions
        if first[1].casefold() == second[1].casefold():
            raise ValueError('Two distinct independent reviewers are required')
        if first[0] != second[0]:
            disagreements.append(key)
            continue
        if first[0]['review_label'] in ('uncertain', 'unreviewed'):
            continue
        clip = clips[key]
        original = dict(first[0], reviewer_1=first[1], reviewer_2=second[1],
            source_video=clip['source_video'], independent_human_review=True,
            evidence_record_ids=[first[2], second[2]],
            agreement_policy='Exact semantic agreement; raw notes preserved in evidence records')
        record = normalize(original, by_id[clip['video_id']], f'human_agreement:{pack_hash}:{key}', 'manual_json')
        if record['review_status'] != 'independently_verified' or record['issues']:
            raise ValueError('Agreed evidence failed annotation validation')
        added.append(record)
        agreed.append(key)
    records = list(base['records']) + added
    report = coverage_report(records)
    report.update(added_records=len(added), agreed_clips=agreed,
        disagreement_clips=disagreements, single_review_clips=[k for k, v in votes.items() if len(v) == 1],
        unreviewed_clips=[k for k in clips if k not in votes],
        trained_model=False, verification_policy='Two attested humans with exact matching label, coverage, cue/contact/fight bounds. IDs do not prove independence.')
    return records, report


def import_forms(args):
    rows = load_locked(args.locked)
    pack = json.loads(Path(args.pack).read_text(encoding='utf-8'))
    base = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    expected = sha256(args.locked)
    if base.get('schema') != 'c3_review_evidence_v1' or any(d.get('locked_manifest_sha256') != expected for d in (pack, base)):
        raise ValueError('Pack/base annotations differ from locked splits')
    if pack.get('base_annotations_sha256') != sha256(args.annotations):
        raise ValueError('Base evidence changed; prepare a new review version')
    pack_root = Path(args.pack).resolve().parent
    for clip in pack['clips']:
        media = (pack_root/clip['media']).resolve()
        if not media.is_relative_to(pack_root) or sha256(media) != clip['media_sha256']:
            raise ValueError('Review media changed or escaped its pack directory')
    forms = [(str(Path(p).resolve()), json.loads(Path(p).read_text(encoding='utf-8-sig'))) for p in args.reviews]
    records, report = import_evidence(pack, sha256(args.pack), rows, base, forms)
    doc = dict(base, records=records, parent_annotations_sha256=sha256(args.annotations),
        source_sha256=dict(base.get('source_sha256', {}), **{str(Path(p).resolve()): sha256(p) for p in args.reviews}),
        human_review_pack_sha256=sha256(args.pack))
    out = Path(args.out_dir)
    write_json(out/'annotations.json', doc)
    write_json(out/'annotation_coverage.json', report)
    print(json.dumps({k: report[k] for k in ('added_records', 'agreed_clips', 'disagreement_clips',
        'single_review_clips', 'unreviewed_clips', 'verified_precursor_videos_by_split')}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'import'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
        cmd.add_argument('--annotations', default='data/foundation/v1/annotations.json')
        if name == 'prepare':
            cmd.add_argument('--train-index', default='data/pose_cache/quick_v1/pilot_train_index.json')
            cmd.add_argument('--val-index', default='data/pose_cache/quick_v1/pilot_val_index.json')
            cmd.add_argument('--out-dir', default='data/reviewed_annotations/quick_human_v1')
            cmd.add_argument('--ffmpeg', default='ffmpeg')
            cmd.set_defaults(run=prepare)
        else:
            cmd.add_argument('--pack', default='data/reviewed_annotations/quick_human_v1/pack.json')
            cmd.add_argument('--reviews', nargs='+', required=True)
            cmd.add_argument('--out-dir', default='data/foundation/reviews_human_v2')
            cmd.set_defaults(run=import_forms)
    args = p.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
