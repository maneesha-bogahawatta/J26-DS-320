"""Build a local final research dashboard without remote scripts or uploads."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

from src.foundation_io import write_json, write_new, sha256


def build(args):
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = {name: json.loads(Path(path).read_text()) for name,path in
        (('Baseline','outputs/advanced_v1/baseline_guarded/evaluation.json'),
         ('Advanced','outputs/advanced_v1/advanced_guarded/evaluation.json'))}
    os.environ['MPLCONFIGDIR'] = str((root/'matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Weak first-onset forecast', 'Weak active fight']
    heads = ['weak_first_onset_within_3s','weak_active_fight']
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax, measure, title in zip(axes, ['auroc','f1'], ['Ranking AUROC','F1 at train-dev selected threshold']):
        for j,(name, report) in enumerate(reports.items()):
            values = [report['metrics'][head][measure] or 0. for head in heads]
            ax.bar([i+j*.34 for i in range(2)],values,width=.32,label=name,color=['#7f8b9a','#158f91'][j])
        ax.set_xticks([.17,1.17],names,fontsize=9);ax.set_ylim(0,1);ax.set_title(title,fontsize=11)
        if measure=='auroc':ax.axhline(.5,color='#a33',linestyle='--',linewidth=1,label='Chance ranking')
        ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same expanded diagnostic validation contexts · original weak UBI targets',fontsize=12)
    figure = root/'model_comparison.png'
    if figure.exists(): raise FileExistsError('Comparison figure already exists')
    fig.savefig(figure,dpi=160);plt.close(fig)
    table = []
    for name, report in reports.items():
        for head, title in zip(heads,names):
            m=report['metrics'][head]
            table.append(f'<tr><td>{name}</td><td>{title}</td><td>{m["n"]}</td><td>{m["auroc"]:.3f}</td><td>{m["precision"]:.1%}</td><td>{m["recall"]:.1%}</td><td>{m["f1"]:.3f}</td></tr>')
    demonstrations = []
    traces = []
    for folder in sorted(root.glob('demo_*')):
        path=folder/'prediction.json'
        if not path.exists():continue
        prediction=json.loads(path.read_text())
        with (folder/'scores.csv').open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        traces.append(dict(id=folder.name,start=prediction['source_context'][0],end=prediction['source_context'][1],rows=[dict(t=float(r['t_s']),pre=float(r['weak_onset_score']) if r['weak_onset_score'] else None,
            fight=float(r['fight_score']) if r['fight_score'] else None) for r in rows]))
        demonstrations.append(f'<article><h3>{html.escape(prediction["video_id"])}</h3><video id="{folder.name}" controls preload="metadata" src="{folder.name}/demo.mp4"></video><canvas data-video="{folder.name}" width="900" height="180"></canvas><p>Original context: {prediction["source_context"][0]:.2f}–{prediction["source_context"][1]:.2f}s · Usable decisions: {prediction["timing"]["valid_decisions"]}/{prediction["timing"]["decisions"]}</p><a href="{folder.name}/scores.csv">Scores CSV</a> · <a href="{folder.name}/prediction.json">Provenance and timing</a></article>')
    selected=json.loads(Path('outputs/advanced_v1/deployment_selection.json').read_text())
    runtime=json.loads((root/'runtime_evaluation.json').read_text())
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Component 03 · Research output</title>
<style>body{font:16px system-ui;background:#101d27;color:#eaf2f7;margin:0;padding:32px;max-width:1180px;margin:auto}h1{font-size:32px}p{line-height:1.6;color:#c9d9e3}a{color:#8edadd}.notice,article,.card{background:#1b303e;border:1px solid #39515f;padding:20px;border-radius:12px;margin:20px 0}.notice{border-color:#b68b3a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #39515f;text-align:left}img{width:100%;border-radius:8px}video{width:100%;max-height:520px;background:black}canvas{width:100%;background:#12232e}code{color:#9fe6dd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:15px}.grid .card{margin:0}strong{color:#fff}</style>
<h1>Component 03 · Pose-based research pipeline</h1>
<p>Locked data → causal YOLO11-Pose/ByteTrack cache → ST-GCN → quality-aware cached replay.</p>
<div class="notice"><strong>Weak supervision · research scores · operational alerts disabled</strong><p>The forecast target is an upcoming original UBI first annotated fight onset within 3 seconds. No independent human precursor labels were created. Scores are uncalibrated, validation uses selected diagnostic contexts, and full-recording/test/real-time performance is not established.</p></div>
<div class="grid"><div class="card"><strong>115 training sources</strong><p>92 fit / 23 internal development</p></div><div class="card"><strong>13 usable validation sources</strong><p>16 indexed contexts; quality abstention retained</p></div><div class="card"><strong>0 test sources used</strong><p>Locked test identities preserved</p></div></div>
<div class="card"><h2>Held-out weak-label diagnostics</h2><p>Both model variants use the same expanded data and first-onset target policy. Checkpoint and threshold selection use internal development sources from the locked training split. The earlier eight-source pilot was historically inspected; newly added sources are reported separately in the evaluation JSON.</p><img src="model_comparison.png" alt="Baseline and advanced model comparison"><table><thead><tr><th>Model</th><th>Weak target</th><th>Windows</th><th>AUROC</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>__TABLE__</tbody></table><p>Default model: <strong>__MODEL__</strong>. The advanced candidate had lower internal development BCE but failed the predeclared forecast research gate. Default selection prefers a model passing that internal gate; validation scores were not used in the selection function. Operational alerts remain disabled. Overlapping windows are not independent events.</p></div>
<div class="notice"><strong>Final current-observation replay: __COVERAGE__ usable coverage</strong><p>With research alerts explicitly opted in, only __HITS__ of __EVENTS__ weak annotated first-onset contexts were detected; __FALSE__ alerts occurred across weak no-fight contexts. This is insufficient for operational use. The demo defaults to scores only. Missing poses produce abstention, never a negative prediction. <a href="runtime_evaluation.json">Full runtime diagnostic</a></p></div>
<h2>Causal replay demonstrations</h2><p>Green joints are observed; orange joints are causally imputed. Low-quality poses or fewer than six currently observed joints in any person produce abstention. The plots show raw uncalibrated onset (teal) and fight (orange) scores. Cached replay timing excludes pose extraction. Click a plot to seek the video.</p>
__VIDEOS__
<div class="card"><h2>Run your own video</h2><p>Use <code>RUN_C3_DEMO.ps1 -Video 'C:\\path\\video.mp4' -Seconds 8</code> from the component folder, or the exact Python commands in <code>FINAL_C3.md</code>. Results are written to a new directory. CPU extraction may be slower than the source video. No files are uploaded.</p></div>
<script>const traces=__TRACES__;for(const trace of traces){const canvas=document.querySelector(`[data-video="${trace.id}"]`),video=document.getElementById(trace.id),ctx=canvas.getContext('2d'),rows=trace.rows,start=trace.start,end=trace.end;function draw(){ctx.clearRect(0,0,900,180);ctx.strokeStyle='#48606d';for(const y of [20,90,160]){ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(880,y);ctx.stroke()}for(const [field,color]of [['pre','#64dbd3'],['fight','#ffbe76']]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let active=false;for(const row of rows){if(row[field]===null){active=false;continue}const x=40+840*(row.t-start)/(end-start),y=160-140*row[field];active?ctx.lineTo(x,y):ctx.moveTo(x,y);active=true}ctx.stroke()}ctx.fillStyle='#d5e4ed';ctx.font='13px sans-serif';ctx.fillText('1.0',4,25);ctx.fillText('0.0',4,164);ctx.fillText(start.toFixed(1)+'s',40,177);ctx.fillText(end.toFixed(1)+'s',820,177);ctx.strokeStyle='#fff';const x=40+840*video.currentTime/(end-start);ctx.beginPath();ctx.moveTo(x,15);ctx.lineTo(x,162);ctx.stroke()}video.ontimeupdate=draw;canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*900/rect.width;video.currentTime=Math.max(0,Math.min(end-start,(x-40)/840*(end-start)))};draw()}</script></html>'''
    text=text.replace('__TABLE__',''.join(table)).replace('__MODEL__',html.escape(selected['variant']))
    text=text.replace('__COVERAGE__',f'{runtime["usable_coverage"]:.1%}').replace('__HITS__',str(runtime['weak_event_contexts_detected'])).replace('__EVENTS__',str(runtime['weak_event_contexts_total'])).replace('__FALSE__',str(runtime['weak_negative_context_alerts']))
    text=text.replace('__VIDEOS__',''.join(demonstrations)).replace('__TRACES__',json.dumps(traces).replace('<','\\u003c'))
    write_new(root/'index.html',text)
    write_json(root/'report_manifest.json',dict(selected_model=selected,comparison_sha256=sha256(figure),
        demos=[r['id'] for r in traces],external_assets=False,verified_early_aggression_claim=False))
    print('Final local report:',(root/'index.html').resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',default='outputs/final_c3_v2')
    build(p.parse_args())
