import csv,json
import numpy as np
import pytest
from src.labels import targets
from src.features import audio_feature,visual_feature
from src.audio_events import CUE_NAMES,cue_report
from src.manifest import split_groups
from src.ubi_annotations import binary_to_intervals,load_annotation
from src.review_annotations import finalize
from src.evaluate_events import debounce,score_events,eval_at,calibrate_val

def test_pre_onset_only_when_verified_cue():
    t=np.arange(1,11,dtype=np.float32)
    y=targets(t,[[7.,9.]],horizon=3,precursors=[[4.4,6.]])
    assert y['pre'].tolist()==[0,0,0,0,1,1,0,0,0,0]
    assert y['pre_mask'][3]==0 and y['pre_mask'][4]==1
    assert y['fight'][6:9].tolist()==[1,1,0]
    unlabeled=targets(t,[[7,9]],horizon=3)
    assert unlabeled['pre'].sum()==0
    assert unlabeled['pre_mask'][3:6].sum()==0

def test_feature_shapes_and_silence():
    assert audio_feature(np.zeros(16000)).shape==(16,)
    assert np.isfinite(audio_feature(np.random.randn(16000))).all()
    assert visual_feature(np.zeros((80,100,3),np.uint8)).shape==(16,)
    assert cue_report(np.zeros(24))["screaming"]==0
    assert len(CUE_NAMES)==8

def test_group_disjoint_and_locked_splits():
    rows=[{'group':f'g{i//2}','video':str(i)} for i in range(30)]
    rows[0]['split']=rows[1]['split']='test'
    splits=split_groups(rows)
    keys=[set(x['group'] for x in s) for s in splits]
    assert not any(keys[i]&keys[j] for i in range(3) for j in range(i+1,3))
    assert 'g0' in keys[2]
    rows[1]['split']='train'
    with pytest.raises(ValueError,match='multiple locked splits'):split_groups(rows)

def test_ubi_label_binary_converts_true_contact_frame():
    lab=[0,0,1,1,0,1,0]
    assert binary_to_intervals(lab,2)==[[1.,2.],[2.5,3.]]

def test_ubi_load_annotation_never_guesses_format(tmp_path):
    labels=tmp_path/'F_1.txt';labels.write_text('0\n0\n1\n1\n0\n')
    assert load_annotation(labels,'frame_binary',5,5)==[[.4,.8]]
    with pytest.raises(ValueError,match='count mismatch'):load_annotation(labels,'frame_binary',20,5)
    labels.write_text('4 8\n12 15\n')
    assert load_annotation(labels,'interval_frames',30,5)==[[.8,1.6],[2.4,3.]]

def test_review_requires_independent_review_and_real_precursor(tmp_path):
    manifest=tmp_path/'candidate.csv';out=tmp_path/'verified.csv'
    cols=['video','audio','source','group','intervals','annotation_status','has_audio','split','duration_s',
          'source_annotation','reviewer_a','reviewer_b','decision','onset_override_s','precursors','notes','audio_cue_review']
    base=dict(video='test.mp4',audio='',source='ubi_fights',group='g0',intervals='[[7,9]]',annotation_status='weak',
       has_audio='false',split='train',duration_s='10',source_annotation='gt.txt',reviewer_a='X',reviewer_b='X',
       decision='approve',onset_override_s='',precursors='[[4,6]]',notes='',audio_cue_review='')
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerow(base)
    with pytest.raises(ValueError,match='No approved'):finalize(manifest,out)
    base['reviewer_b']='Y'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerow(base)
    accepted,bad=finalize(manifest,out)
    assert len(accepted)==1 and not bad
    r=next(csv.DictReader(out.open()))
    assert r['annotation_status']=='verified' and json.loads(r['precursors'])==[[4,6]]

def test_debouncer_needs_current_vote_and_ignores_low_quality():
    t=[1,2,3,4,5]
    assert debounce(t,[.7,.7,.1,.1,.1],threshold=.5)==[]
    assert len(debounce(t,[.7,.7,.7,.1,.1],threshold=.5))==1
    assert debounce(t,[.7]*5,[.2]*5,threshold=.5)==[]

def test_event_recall_lead_time_and_false_alerts_are_separate():
    rows=[dict(video='pos',times=list(range(1,21)),duration_s=20,intervals=[[10,13]],
               alerts=[{'timestamp_s':7,'risk':.8},{'timestamp_s':11,'risk':.9}]),
          dict(video='normal',times=list(range(1,21)),duration_s=20,intervals=[],
               alerts=[{'timestamp_s':6,'risk':.8}])]
    e=score_events(rows)
    assert e['fight_onsets']==1 and e['early_detected_onsets']==1
    assert e['median_lead_seconds']==3.
    assert e['early_lead_recall']['3']['recall']==1.
    assert e['late_alerts_during_fights']==1
    assert e['false_early_alerts']==1
    assert round(e['false_alerts_per_camera_hour'])==90

def test_validation_calibration_only_available_with_positive_negative():
    rows=[dict(video='p',times=[1,2,3,4,5,6,7,8,9,10],scores=[.1,.1,.9,.9,.9,.1,.1,.1,.1,.1],
               quality=[1]*10,intervals=[[6,8]],duration_s=10),
          dict(video='n',times=[1,2,3,4,5,6,7,8,9,10],scores=[.1]*10,
               quality=[1]*10,intervals=[],duration_s=10)]
    out=calibrate_val(rows,max_false_per_hour=10)
    assert out['status']=='val_target_met'
    assert out['threshold_watch']>=.2
