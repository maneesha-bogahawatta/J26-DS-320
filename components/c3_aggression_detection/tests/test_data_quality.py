"""Synthetic fixtures only; no example here is exported as research ground truth."""
import copy
import unittest

import numpy as np

from src.contact_review import CONTACT_DEFINITION, FORM_SCHEMA, import_contact_evidence
from src.data_quality import evidence_audit, select_balanced, strict_records
from src.evaluate_verified import evaluate, subtract_ranges, union_ranges, validate_timeline, window_report
from src.human_review import PACK_SCHEMA


class DataQualityTests(unittest.TestCase):
    def fixture(self,label='precursor',key='p'):
        row=dict(video_id=key,video=key+'.mp4',split='val',duration_s='10',content_sha256=key,leakage_group=key,intervals='[[8,10]]')
        clip=dict(clip_id=key,video_id=key,split='val',source_sha256=key,leakage_group=key,clip_start_s=0.,clip_end_s=10.)
        raw=dict(clip_id=key,review_label=label,timestamp_basis='original_video_seconds',reviewed_ranges_s=[[0.,10.]],
            precursors=[[6.,7.]],first_threatening_cue_s=6.,first_physical_contact_s=8.,verified_fight_intervals=[[8.,10.]],
            contact_observation='confirmed_visible',contact_uncertainty_s=None,contact_definition=CONTACT_DEFINITION,
            contact_personally_observed=True,contact_not_inferred_from_dataset_onset=True)
        if label=='non_aggressive':raw.update(precursors=[],first_threatening_cue_s=None,first_physical_contact_s=None,
            verified_fight_intervals=[],contact_observation='not_observed',contact_personally_observed=False,
            contact_not_inferred_from_dataset_onset=False)
        forms=[(key+'_'+v,dict(schema=FORM_SCHEMA,pack_sha256='hash',split='val',reviewer=v,
            independent_human_review=True,records=[copy.deepcopy(raw)])) for v in ('SyntheticAlice','SyntheticBob')]
        records,_=import_contact_evidence(dict(schema=PACK_SCHEMA,clips=[clip]),'hash',[row],dict(records=[]),forms)
        return dict(records=records)

    def records(self):
        a=self.fixture();b=self.fixture('non_aggressive','n');all_records=a['records']+b['records']
        accepted,rejected=strict_records(dict(records=all_records));self.assertFalse(rejected)
        return accepted

    def scores(self,early=True,abstain=False):
        rows=[]
        for key in ('p','n'):
            for t in np.arange(0,10,.2):
                high=(6<=t<7 if early else 8<=t<9) if key=='p' else 2<=t<3
                valid=not(abstain and key=='p')
                rows.append(dict(video_id=key,t_s=float(t),weak_onset_score=.9 if high else .1,
                    fight_score=.1,valid_pose=valid,segment_id=0,track_ids_json='[1,2]'))
        return rows

    def test_two_independent_raw_votes_required_for_readiness(self):
        doc=self.fixture();accepted,rejected=strict_records(doc)
        self.assertEqual(len(accepted),1);self.assertEqual(rejected,[])
        doc['records'][0]['original']['independent_human_review']=False
        self.assertEqual(strict_records(doc)[0],[])
        self.assertEqual(len(strict_records(doc)[1]),1)

    def test_missing_vote_forged_consensus_and_legacy_are_not_verified(self):
        for action in ('missing','forged','legacy'):
            doc=self.fixture()
            if action=='missing':doc['records'].pop(0)
            elif action=='forged':doc['records'][-1]['reviewers']=['Same','same']
            else:doc['records'][-1].pop('source_schema')
            self.assertFalse(strict_records(doc)[0])

    def test_uncertain_ai_and_weak_stay_separate(self):
        base=self.fixture()['records'][0]
        records=[dict(base,record_id='weak',review_status='weak',label='weak'),
            dict(base,record_id='ai',review_status='provisional_ai_review',label='precursor'),
            dict(base,record_id='uncertain',review_status='provisional_single_review',label='uncertain')]
        report,partitions=evidence_audit(dict(records=records))
        self.assertEqual(report['partition_counts'],dict(weak=1,provisional=1,uncertain=1))
        self.assertEqual(report['strict_verified']['val']['precursor_positive_videos'],0)
        self.assertEqual(partitions['provisional'][0]['review_status'],'provisional_ai_review')

    def test_test_consensus_never_counts_toward_development(self):
        doc=self.fixture()
        for r in doc['records']:r['split']='test'
        self.assertEqual(strict_records(doc),([],[]))
        with self.assertRaises(ValueError):evaluate([],[],{},[.5,.5],[True,False],split='test')

    def test_balanced_source_disjoint_selection_is_reproducible(self):
        pool=[dict(video_id=f'{s}_{k}_{i}',split=s,weak_stratum=k,leakage_group=f'{s}_{k}_{i}',
            usable_decisions=30,priority=i) for s in ('train','val') for k in ('weak_fight','weak_no_fight') for i in range(8)]
        first,shortfalls=select_balanced(pool,dict(train=4,val=2));second,_=select_balanced(list(reversed(pool)),dict(train=4,val=2))
        self.assertFalse(shortfalls);self.assertEqual(first,second)
        self.assertEqual(len({c['leakage_group'] for c in first}),12)
        self.assertEqual(len([c for c in first if c['split']=='train']),8)

    def test_pose_failures_are_deferred_and_shortfalls_explicit(self):
        pool=[dict(video_id='x',split='train',weak_stratum='weak_fight',leakage_group='x',usable_decisions=0,priority=100)]
        chosen,shortfalls=select_balanced(pool,dict(train=1,val=1))
        self.assertFalse(chosen);self.assertEqual(len(shortfalls),4)

    def test_interval_exposure_union_and_conflict_subtraction(self):
        self.assertEqual(union_ranges([[0,4],[2,6],[6,10]]),[[0.,10.]])
        self.assertEqual(subtract_ranges([[0,10]],[[2,4],[6,8]]),[[0.,2.],[4.,6.],[8.,10.]])

    def test_incomplete_duplicate_or_shifted_timeline_rejected(self):
        validate_timeline([dict(t_s=0),dict(t_s=.2)],[0,.2])
        for rows in ([dict(t_s=0)],[dict(t_s=0),dict(t_s=0)],[dict(t_s=.1),dict(t_s=.3)]):
            with self.assertRaises(ValueError):validate_timeline(rows,[0,.2])

    def test_no_verified_data_has_null_research_metrics(self):
        result=evaluate([],self.scores(),dict(p=[0,10],n=[0,10]),[.5,.5],[True,False])
        self.assertEqual(result['status'],'blocked');self.assertIsNone(result['event_recall'])
        self.assertIsNone(result['heads']['verified_precursor']['false_alarms_per_reviewed_negative_hour'])
        self.assertIsNone(result['heads']['verified_precursor']['pr_auc_trapezoid'])

    def test_verified_lead_precision_recall_and_negative_hours(self):
        result=evaluate(self.records(),self.scores(),dict(p=[0,10],n=[0,10]),[.5,.5],[True,False])
        self.assertEqual(result['eligible_contact_events'],1);self.assertEqual(result['true_positive_events'],1)
        self.assertEqual(result['event_recall'],1);self.assertEqual(result['event_precision'],.5)
        self.assertAlmostEqual(result['event_f1'],2/3);self.assertAlmostEqual(result['median_lead_seconds'],1.8)
        early=result['heads']['verified_precursor']
        self.assertAlmostEqual(early['reviewed_negative_camera_hours'],10/3600)
        self.assertAlmostEqual(early['false_alarms_per_reviewed_negative_hour'],360)

    def test_duplicate_negative_coverage_does_not_double_hours(self):
        records=self.records();records.append(copy.deepcopy(records[-1]))
        result=evaluate(records,self.scores(),dict(p=[0,10],n=[0,10]),[.5,.5],[True,False])
        self.assertAlmostEqual(result['heads']['verified_precursor']['reviewed_negative_camera_hours'],10/3600)

    def test_no_pose_event_stays_in_recall_denominator(self):
        result=evaluate(self.records(),self.scores(abstain=True),dict(p=[0,10],n=[0,10]),[.5,.5],[True,False])
        self.assertEqual(result['false_negative_events'],1);self.assertEqual(result['event_recall'],0)
        self.assertEqual(result['heads']['verified_precursor']['pose_abstentions'],5)

    def test_late_alert_never_counts_as_early_true_positive(self):
        result=evaluate(self.records(),self.scores(early=False),dict(p=[0,10],n=[0,10]),[.5,.5],[True,False])
        self.assertEqual(result['true_positive_events'],0);self.assertEqual(result['late_early_head_alerts'],1)

    def test_disabled_head_does_not_emit_even_when_scores_are_high(self):
        result=evaluate(self.records(),self.scores(),dict(p=[0,10],n=[0,10]),[.5,.5],[False,False])
        self.assertEqual(result['true_positive_events'],0)
        self.assertEqual(result['heads']['verified_precursor']['false_alerts_on_reviewed_negative_time'],0)

    def test_single_class_ranking_remains_undefined(self):
        result=window_report([0,0],[.1,.2],.5)
        self.assertIsNone(result['average_precision']);self.assertIsNone(result['recall'])


if __name__=='__main__':unittest.main()
