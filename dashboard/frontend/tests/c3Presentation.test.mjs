import assert from 'node:assert/strict';
import test from 'node:test';
import { annotatedContext, booleanEvidence, finiteScore, headDiagnostics, mediaOffset, scoreSegments } from '../src/lib/c3Presentation.ts';

test('completed rendering is playable even without any violence candidate or usable scores', () => {
  const job = { status: 'completed', files: ['demo.mp4', 'scores.csv'], events: [],
    prediction: { source_context: [13.4, 21.4], timing: { valid_decisions: 0, decisions: 40 } } };
  assert.deepEqual(annotatedContext(job), { start: 13.4, end: 21.4 });
  assert.ok(Math.abs(mediaOffset(17.8, annotatedContext(job).start) - 4.4) < 1e-8);
});

test('failed, pending or missing rendered artifacts never replace the original upload', () => {
  const job = { status: 'completed', files: ['demo.mp4'], prediction: { source_context: [0, 8] } };
  for (const status of ['failed', 'running', 'queued', 'interrupted']) assert.equal(annotatedContext({ ...job, status }), null);
  assert.equal(annotatedContext({ ...job, files: ['scores.csv'] }), null);
  assert.equal(annotatedContext(undefined), null);
  for (const range of [[0, 0], [3, 2], [-1, 3], [0, NaN], [0, Infinity]]) {
    assert.equal(annotatedContext({ ...job, prediction: { source_context: range } }), null);
  }
});

test('missing or invalid model score remains absent instead of becoming zero', () => {
  for (const value of [null, undefined, '', 'NaN', Infinity, -0.1, 1.1]) assert.equal(finiteScore(value), null);
  assert.equal(finiteScore('0'), 0);
  assert.equal(finiteScore('0.75'), .75);
});

test('chart breaks its line at no-pose scores', () => {
  const rows = [
    { t_s: 10, weak_onset_score: .1, fight_score: .2 },
    { t_s: 11, weak_onset_score: '', fight_score: null },
    { t_s: 12, weak_onset_score: .9, fight_score: .8 },
  ];
  const segments = scoreSegments(rows, 'weak_onset_score', 10, 12);
  assert.equal(segments.length, 2);
  assert.equal(segments[0], '42,156');
  assert.equal(segments[1], '878,44');
  assert.deepEqual(scoreSegments(rows, 'fight_score', 10, 10), []);
});

test('source-video time is converted to annotated-context time when seeking', () => {
  assert.ok(Math.abs(mediaOffset(17.833333, 13.433333) - 4.4) < 1e-8);
  assert.equal(mediaOffset(1, 4), 0);
  assert.equal(mediaOffset(NaN, 4), 0);
});

test('explicit boolean evidence does not treat the string False as true', () => {
  assert.equal(booleanEvidence('False'), false);
  assert.equal(booleanEvidence('false'), false);
  assert.equal(booleanEvidence('True'), true);
  assert.equal(booleanEvidence(true), true);
});

test('head diagnostics distinguish raw score, threshold, disabled candidate policy and missing pose', () => {
  const heads = headDiagnostics({prediction:{thresholds:[.604,.611],research_alert_heads_enabled:[true,false]},scores:[
    {t_s:1,weak_onset_score:.52,fight_score:.55,valid_pose:'True'},
    {t_s:2,weak_onset_score:1,fight_score:1,valid_pose:'False'},
    {t_s:3,weak_onset_score:'',fight_score:null,valid_pose:true},
  ]});
  assert.equal(heads[0].peak,.52); assert.equal(heads[0].candidateEnabled,true);
  assert.equal(heads[1].peak,.55); assert.equal(heads[1].threshold,.611); assert.equal(heads[1].candidateEnabled,false);
  for (const head of headDiagnostics({})) { assert.equal(head.peak,null); assert.equal(head.threshold,null); assert.equal(head.candidateEnabled,null); }
});
