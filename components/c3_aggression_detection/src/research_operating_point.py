"""Select conservative research operating points using internal development only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import (SCHEMA, Windows, build_model, load_windows, metrics, score)
from src.foundation_io import sha256, write_json


def select_point(y, probabilities, max_fpr=.10):
    y = np.asarray(y, int); p = np.asarray(probabilities, float)
    if not len(y) or len(set(y.tolist())) != 2:
        raise ValueError('Development operating point needs both weak classes')
    candidates = np.unique(np.r_[p, np.nextafter(p, np.inf), 1.000001])
    allowed = []
    for threshold in candidates:
        report = metrics(y, p, float(threshold))
        fpr = report['fp']/max(1, report['fp']+report['tn'])
        if fpr <= max_fpr+1e-12:
            allowed.append((report, fpr))
    report, fpr = max(allowed, key=lambda value: (value[0]['f1'], value[0]['recall'], value[0]['threshold']))
    enabled = bool(report['recall'] >= .20 and report['balanced_accuracy'] >= .55)
    return dict(metrics=report, development_fpr=fpr, research_alert_eligible=enabled)


def run(args):
    torch.set_num_threads(2)
    data, meta = load_windows(args.data)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA or checkpoint['windows_sha256'] != meta['windows_sha256']:
        raise ValueError('Checkpoint/window provenance mismatch')
    model = build_model(checkpoint['variant'])
    model.load_state_dict(checkpoint['state'], strict=True)
    dev = Windows(data, 'dev')
    scored = score(model, DataLoader(dev, batch_size=32), checkpoint['variant'])
    results = []
    for h in range(2):
        mask = scored['masks'][:, h].numpy()
        results.append(select_point(scored['targets'][:, h].numpy()[mask],
                                    scored['logits'][:, h].sigmoid().numpy()[mask]))
    policy = dict(source='locked train internal development only', max_development_fpr=.10,
        minimum_development_recall=.20, minimum_development_balanced_accuracy=.55,
        external_validation_used=False, population_false_alarm_rate_claim=False)
    checkpoint.update(thresholds=[r['metrics']['threshold'] for r in results],
        research_alert_heads_enabled=[r['research_alert_eligible'] for r in results],
        operating_point_policy=policy, source_checkpoint_sha256=sha256(args.checkpoint),
        operational_alerts_enabled=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open('xb') as stream: torch.save(checkpoint, stream)
    report = dict(policy=policy, heads=dict(zip(meta['target_heads'], results)),
        checkpoint_sha256=sha256(args.out), model_variant=checkpoint['variant'],
        note='Development FPR is on sampled weak labels, not full-recording false alarms/hour or verified accuracy.')
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/pose_cache/advanced_v1/windows')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    run(p.parse_args())
