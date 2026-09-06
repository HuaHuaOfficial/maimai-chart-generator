"""Publish only the exact immutable drafts accepted by their Harness sessions."""
from __future__ import annotations
from pathlib import Path
import json
from .simai import render_compact_maidata,parse_maidata,parse_inote_ticks


def publish(prepared,results,codecs,folder):
    from ..app.preparation import _write_track_mp3
    folder=Path(folder);lines=[f"&title={prepared['title']}",f"&artist={prepared['metadata']['artist']}",f"&first={prepared['beat_offset']:g}",f"&wholebpm={prepared['bpm']:g}",f"&versionid={prepared['version_id']}",f"&version={prepared['version_name']}",'&clock_count=4','&chartgenerator=ChartRuntime-0.3.1','']
    records=[]
    for slot,entry in sorted(results.items()):
        result,backend,generator,request=entry;chart=result.chart;permit=result.permit
        if result.state!='accepted' or chart is None or permit is None:raise RuntimeError('Unaccepted session cannot publish')
        if permit.chart!=chart.ref or permit.definition!=request.definition:raise RuntimeError('Publish permit belongs to another chart')
        evaluation=backend.known[permit.receipt_id]
        if backend.permit(evaluation)!=permit:raise RuntimeError('Publish permit has no matching Harness receipt')
        chart.payload.assert_unmodified();events=chart.payload.events
        text=render_compact_maidata(title=prepared['title'],source_name='track.mp3',version_name=prepared['version_name'],version_id=prepared['version_id'],difficulty_slot=slot,internal_level=prepared['levels'][slot],bpm=prepared['bpm'],events=events,total_ticks=prepared['total_ticks'],bpm_changes=dict(zip(map(int,prepared['bpm_ticks']),map(float,prepared['bpm_values']))),first=prepared['beat_offset'])
        inote=parse_maidata(text)[f'inote_{slot}'];parsed=parse_inote_ticks(inote,prepared['bpm'])
        replay=codecs[slot].encode(dict(parsed.events),parsed.bpm_ticks,parsed.bpm_values)
        if replay.digest!=chart.ref.content_digest:raise RuntimeError('Simai encoding changed the accepted IR')
        lines.extend((f'&lv_{slot}={prepared["levels"][slot]:.1f}',f'&des_{slot}=ChartRuntime {prepared["spec"].label}',f'&inote_{slot}={inote}',''))
        meta=backend.results[chart.ref]
        records.append({'difficultySlot':slot,'internalLevel':prepared['levels'][slot],'events':len(events),'contentDigest':chart.ref.content_digest,'receiptId':permit.receipt_id,
                        'definition':vars(request.definition),'harness':meta,'generationPhases':generator.timings,'feedbackRounds':len(result.observations),
                        'architectureActors':['generator','harness']})
    folder.mkdir(parents=True,exist_ok=True)
    pending=folder/'maidata.pending.txt';pending.write_text('\n'.join(lines),encoding='utf8')
    audio_pending=folder/'track.pending.mp3';_write_track_mp3(prepared['audio_path'],audio_pending,prepared['ffmpeg'])
    audio_pending.replace(folder/'track.mp3')
    document={'schemaVersion':4,'release':'0.3.1','title':prepared['title'],'versionId':prepared['version_id'],'versionName':prepared['version_name'],'bpm':prepared['bpm'],'first':prepared['beat_offset'],
              'levels':{str(k):v for k,v in prepared['levels'].items()},'audioDurationSeconds':prepared['duration'],'roundedTotalTicks':prepared['total_ticks'],'endSeconds':prepared['end_seconds'],
              'charts':records,'timings':prepared['timings'],'outputDir':str(folder),'inferenceBackend':prepared['acceleration_info'],'cpuMusicalChecks':False}
    (folder/'metadata.pending.json').write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'metadata.pending.json').replace(folder/'metadata.json')
    pending.replace(folder/'maidata.txt')
    return document
