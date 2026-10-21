import assert from 'node:assert/strict';
import test from 'node:test';
import { c3Found, checkRecordedVideo, emptyVideoSummary } from '../src/detect/recordedVideo.ts';

const file = { name: 'camera clip.mp4' };
function event() {
  return { event_id:'component_3-job-local', component_id:'component_3', incident_type:'aggression',
    evidence:[{name:'source_timestamp_seconds',value:18.2}], media_ref:'/api/v1/c3/media/job/demo.mp4',
    media_offset_seconds:4.8, bounding_box:{x_min:.1,y_min:.2,x_max:.3,y_max:.4} };
}
function completed(valid = 37, events = [event()]) {
  return {job_id:'saved-job',source_id:'upload:sha',status:'completed',events,
    prediction:{source_context:[0,8],timing:{valid_decisions:valid,decisions:40}}};
}
function dependencies(final = completed()) {
  const calls=[]; let polls=0;
  return {calls, team:async()=>{calls.push('team');return {events:[{event_id:'team-event'}],detectors_run:['c4_fall_and_object']};},
    c3:async(path,options)=>{
      calls.push({path,options});
      if(path.endsWith('/uploads')) return {source_id:'upload:sha'};
      if(path.endsWith('/jobs')) return {job_id:'saved-job',status:'queued'};
      polls++; return polls===1 ? {job_id:'saved-job',status:'running'} : final;
    },wait:async()=>{}};
}

test('ordinary recorded upload runs C3 temporal inference and original registered models',async()=>{
  const deps=dependencies(); const stages=[];let report;
  const result=await checkRecordedVideo(file,{sourceId:'upload-camera',mediaUrl:'blob:local',onProgress:p=>stages.push(p.stage),onReport:r=>{report=r;}},deps);
  assert.equal(deps.calls[0],'team');
  const upload=deps.calls.find(call=>call.path?.endsWith('/uploads'));
  assert.equal(upload.options.body,file);
  assert.equal(upload.options.headers['X-C3-File-Name'],'camera%20clip.mp4');
  const body=JSON.parse(deps.calls.find(call=>call.path?.endsWith('/jobs')).options.body);
  assert.deepEqual(body,{source_id:'upload:sha',camera_id:'upload-camera',seconds:60,research_alerts:false});
  assert.equal(result.teamEvents[0].event_id,'team-event');
  assert.equal(result.c3Found[0].event.component_id,'component_3');
  assert.equal(report.c3.status,'completed');
  assert.ok(stages.some(stage=>stage.includes('C3 running')));
});

test('C3 failure preserves team results but reports an incomplete check',async()=>{
  const deps=dependencies({job_id:'saved-job',status:'failed',error:'pose extraction failed'});
  const result=await checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local'},deps);
  assert.equal(result.teamEvents.length,1);
  assert.deepEqual(result.c3Found,[]);
  assert.equal(emptyVideoSummary(result.report,true),'Video check incomplete');
  assert.match(result.report.c3.error,/pose extraction failed/);
});

test('team detector failure does not discard a valid C3 result',async()=>{
  const deps=dependencies();deps.team=async()=>{throw new Error('team unavailable');};
  const result=await checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local'},deps);
  assert.equal(result.c3Found.length,1);
  assert.equal(result.report.team.status,'failed');
  assert.equal(emptyVideoSummary(result.report,true),'Video check incomplete');
});

test('missing poses produce abstention wording instead of an all-clear',async()=>{
  const result=await checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local'},dependencies(completed(0,[])));
  assert.equal(emptyVideoSummary(result.report,true),'Insufficient pose evidence for C3');
  assert.equal(emptyVideoSummary(null,true),'Video check incomplete');
  assert.equal(emptyVideoSummary({team:{status:'completed',detectors:[]}},false),'No models checked this video');
  assert.equal(emptyVideoSummary({team:{status:'completed',detectors:['c4']}},false),'No candidates returned');
});

test('C3 event media is preserved and its footer box is not drawn over source footage',()=>{
  const original=event();const found=c3Found(original,13.4);
  assert.equal(found.startSeconds,18.2);
  assert.equal(found.event,original);
  assert.equal(found.event.media_ref,'/api/v1/c3/media/job/demo.mp4');
  assert.deepEqual(found.path,[]);
  assert.equal(c3Found({...original,evidence:[]},13.4).startSeconds,18.2);
});

test('cancelling before upload submits no inference and is not shown as success',async()=>{
  const controller=new AbortController();controller.abort();const deps=dependencies();
  await assert.rejects(checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local',signal:controller.signal},deps),{name:'Cancelled'});
  assert.deepEqual(deps.calls,[]);
});

test('both failed checks propagate an error while retaining failure coverage',async()=>{
  const deps=dependencies();deps.team=async()=>{throw new Error('team down');};deps.c3=async()=>{throw new Error('C3 down');};let report;
  await assert.rejects(checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local',onReport:r=>{report=r;}},deps),/Video checks failed/);
  assert.equal(report.team.status,'failed');assert.equal(report.c3.status,'failed');
});

test('a completed job without prediction provenance is not counted as checked',async()=>{
  const deps=dependencies({job_id:'saved-job',status:'completed',events:[]});
  const result=await checkRecordedVideo(file,{sourceId:'camera',mediaUrl:'blob:local'},deps);
  assert.equal(result.report.c3.status,'failed');
  assert.match(result.report.c3.error,/without prediction provenance/);
});
