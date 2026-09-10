"""Frozen training calibration tied to the shared feature definition."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np


@lru_cache(maxsize=128)
def load_calibration(root,version,slot,ds_tenths,bpm):
    # BASIC/ADVANCED never had profiles in this asset. Report that explicitly;
    # trained model conditioning remains available, but no fictitious limits.
    if slot<4:return None
    path=Path(root)/'models/experimental/contextual_calibration.npz'
    if not path.is_file():raise FileNotFoundError(path)
    result=calibrate(root,version,slot,ds_tenths/10.,bpm)
    return {**result,'assetDigest':sha256(path.read_bytes()).hexdigest(),'definition':'training-local-features-v2'}



def calibrate(root,version,slot,ds,bpm,exclude_song_keys=()):
    if not np.isfinite(bpm) or bpm<=0 or not np.isfinite(ds):raise ValueError('Invalid BPM or DS for calibration')
    asset=Path(root)/'models/experimental/contextual_calibration.npz'
    if not asset.is_file():raise RuntimeError(f'Missing offline calibration asset: {asset}')
    with np.load(asset,allow_pickle=False) as data:
        base=(data['slot']==slot)&np.isfinite(data['median_bpm'])&(data['median_bpm']>0)
        if exclude_song_keys:base&=~np.isin(data['song_key'],np.asarray(tuple(exclude_song_keys)))
        distance=np.abs(data['ds'].astype(float)-ds);ratio=data['median_bpm'].astype(float)/bpm
        stages=[('matched',.31,6,.75,1.25),('all_versions',.31,32,.75,1.25),
                ('wider_tempo',.31,32,.5,2.),('same_difficulty_all_tempos',.31,32,0.,float('inf'))]
        attempts=[];ids=np.array([],dtype=int)
        for stage,ds_window,version_window,lo,hi in stages:
            keep=base&(distance<=ds_window)&(np.abs(data['version'].astype(float)-version)<=version_window)&(ratio>=lo)&(ratio<=hi)
            candidates=np.flatnonzero(keep)
            score=10*distance[candidates]+np.abs(data['version'][candidates].astype(float)-version)+5*np.abs(np.log(ratio[candidates]))
            candidates=candidates[np.lexsort((data['song_key'][candidates],score))][:48]
            count=len(set(data['song_key'][candidates].tolist()));attempts.append({'stage':stage,'songs':count})
            if count>=5:ids=candidates;break
        if not len(ids):
            # Sparse DS cells retain the same difficulty slot; never borrow BASIC/ADVANCED profiles.
            stage='nearest_ds_same_slot';candidates=np.flatnonzero(base)
            candidates=candidates[np.lexsort((data['song_key'][candidates],np.abs(np.log(ratio[candidates])),distance[candidates]))]
            ids=candidates[:48]
            if len(set(data['song_key'][ids].tolist()))<5:raise RuntimeError(f'Frozen calibration asset has insufficient profiles for difficulty slot {slot}')
        fallback={'stage':stage,'used':stage!='matched','attempts':attempts,'sameSlot':True,
                  'actualDsRange':[float(data['ds'][ids].min()),float(data['ds'][ids].max())],
                  'actualBpmRange':[float(data['median_bpm'][ids].min()),float(data['median_bpm'][ids].max())]}
        offsets=data['offsets'];all_features=data['features']
        features=[all_features[offsets[i]:offsets[i+1]] for i in ids]
        sources=[{'songKey':str(data['song_key'][i]),'sha256':str(data['source_sha256'][i]),'ds':float(data['ds'][i]),
                  'version':int(data['version'][i]),'events':int(offsets[i+1]-offsets[i])} for i in ids]
    all_rows=np.concatenate(features)
    # Per-feature empirical guardrails; Slide geometry uses a rarer-tail quantile.
    # Floors avoid treating absent mechanisms as zero support.
    thresholds=np.quantile(all_rows,.99,axis=0).tolist()
    slide_rows=all_rows[:,2][all_rows[:,2]>0]
    thresholds[2]=float(np.quantile(slide_rows,.999)) if len(slide_rows) else 1.0
    thresholds=[max(1.,thresholds[0]),max(2.,thresholds[1]),max(1.,thresholds[2]),max(8.,thresholds[3])]
    return {'schemaVersion':2,'target':{'version':version,'slot':slot,'ds':ds,'bpm':bpm},
            'cohortRule':'train, same slot, DS +/-0.3, version +/-6, BPM ratio .75..1.25; nearest <=48 charts; no D8 views',
            'sources':sources,'quantile':.99,'featureQuantiles':[.99,.99,.999,.99],'features':['unpredictabilityWeightedInputs1s','outerKeyStepsPerSecond','slideJudgeAreasPerSecond','unexpectedMotionChangeRate'],
            'cohortFallback':fallback,
            'thresholds':thresholds,
            'tolerance':{'sustainedRatio':1.35,'sustainedEvents':3,'windowSeconds':1.,'extremeRatio':2.,'instantaneousFeatureIndexes':[2]},
            'calibrationAsset':'bundled frozen training-derived numeric profiles; no runtime chart access',
            'scope':'empirical local workload guardrail; not a calibrated local DS estimator; song-section strata pending'}
