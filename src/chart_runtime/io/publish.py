"""Publish only the exact immutable drafts accepted by their Harness sessions."""
from __future__ import annotations
from pathlib import Path
import json
from .simai import render_compact_maidata,parse_maidata,parse_inote_ticks


def publish(prepared,results,codecs,folder,release_folder):
    from ..app.preparation import _write_cover_png,_write_bga_mp4
    from ..app.media_cache import stage_track_mp3
    folder=Path(folder);lines=[f"&title={prepared['title']}",f"&artist={prepared['metadata']['artist']}",f"&first={prepared['beat_offset']:g}",f"&wholebpm={prepared['bpm']:g}",f"&versionid={prepared['version_id']}",f"&version={prepared['version_name']}",'&clock_count=4','&chartgenerator=ChartRuntime-1.1.0','']
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
                        'definition':vars(request.definition),'harness':meta,'generationPhases':generator.timings,'feedbackRounds':max(0,len(result.observations)-1),
                        'architectureActors':['generator','harness']})
    folder.mkdir(parents=True,exist_ok=True)
    pending=folder/'maidata.pending.txt';pending.write_text('\n'.join(lines),encoding='utf8')
    audio_pending=folder/'track.pending.mp3';cached_track,audio_cache_hit=stage_track_mp3(prepared['root'],prepared['audio_path'],audio_pending,prepared['ffmpeg'])
    audio_pending.replace(folder/'track.mp3')
    if prepared['cover_path'] is not None:
        cover_pending=folder/'bg.pending.png';_write_cover_png(prepared['cover_path'],cover_pending,prepared['ffmpeg'])
        cover_pending.replace(folder/'bg.png')
    if prepared['bga_path'] is not None:
        bga_pending=folder/'pv.pending.mp4';_write_bga_mp4(prepared['bga_path'],bga_pending,prepared['ffmpeg'])
        bga_pending.replace(folder/'pv.mp4')
    document={'schemaVersion':4,'release':'1.1.0','title':prepared['title'],'versionId':prepared['version_id'],'versionName':prepared['version_name'],'bpm':prepared['bpm'],'first':prepared['beat_offset'],
              'whatQuotas':{'starScale':float(prepared['metadata']['whatStarScale']),'arity2Scale':float(prepared['metadata']['whatArity2Scale']),'holdScale':float(prepared['metadata']['whatHoldScale']),'touchScale':float(prepared['metadata']['whatTouchScale']),'touchHoldScale':float(prepared['metadata']['whatTouchHoldScale']),'variation':float(prepared['metadata']['whatVariation']),'semantics':'1.0 is the typical official-chart distribution at displayed DS; scales are relative odds in one normalized WHAT configuration distribution, so changing them may also change notes/event and effective difficulty; Stars are never post-filled'},
              'levels':{str(k):v for k,v in prepared['levels'].items()},'audioDurationSeconds':prepared['duration'],'roundedTotalTicks':prepared['total_ticks'],'endSeconds':prepared['end_seconds'],
              'songId':prepared['song_id'],'songIdAuto':prepared['song_id_auto'],'trackAudio':{'contract':'mp3-320k-v1','bitrateKbps':320,'cacheHitAtPublish':bool(audio_cache_hit)},'charts':records,'timings':prepared['timings'],'outputDir':str(folder),'releaseDir':str(release_folder),'inferenceBackend':prepared['acceleration_info'],'cpuMusicalChecks':False}
    pending.replace(folder/'maidata.txt')
    return document
