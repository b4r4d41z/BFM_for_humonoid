"""Unified executor episodes; generation observations and acceptance-time duration."""
import numpy as np
from bfm.contracts.sharpa import (SCHEMA, PROFILE, FIELDS, GROUPS, JOINTS, MODEL_SHA256,
                                  SharpaV4Contract, flatten_action, flatten_state)
from .execution_v4 import acknowledgements, verify_streams
from .policy_time import policy_timing, continuity_reason


def build(records, metadata, *, max_age_s, collection_commit, max_camera_skew_s=None):
    from .mcap import ACTION, TOPICS, CLOCK, lifecycle, asof, header_ns, decode_image
    from .sharpa import CAMERAS, SOURCES
    bfm = metadata['bfm'];timing = policy_timing(bfm)
    if (bfm.get('contract_version') != 4 or bfm.get('action_dim') != 26
            or bfm.get('action_group_order') != list(GROUPS)
            or metadata.get('camera_configuration',{}).get('model_sha256') != MODEL_SHA256
            or metadata.get('recording_code',{}).get('commit_sha') != collection_commit
            or len(collection_commit)!=40 or any(c not in '0123456789abcdef' for c in collection_commit) or metadata.get('time',{}).get('synchronization_source')!='rosbag_record_timestamp'):
        raise ValueError('Invalid v4 source/model/action provenance')
    for samples in records.values(): samples.sort(key=lambda x:x[0])
    actions, config = acknowledgements(records, metadata)
    if not actions:raise ValueError('No v4 actions')
    limits = np.asarray(max_age_s,dtype=float)
    if limits.shape!=(5,) or not np.isfinite(limits).all() or (limits<=0).any(): raise ValueError('Five positive ages required')
    limits = np.rint(limits*1e9).astype(np.int64)
    if max_camera_skew_s is not None and (not np.isfinite(max_camera_skew_s) or max_camera_skew_s<=0): raise ValueError('Invalid camera skew')
    events = lifecycle(records,bfm['episode']['episode_id'],[r for r,_,_,_ in actions])
    ticks=[];rejected=[];names=None;effort=None;encodings={};frames={};sessions=[];boundaries=[]
    previous_clock = None;clock_resets=[]
    for record,clock in records.get(CLOCK,[]):
        value=clock.clock.sec*10**9+clock.clock.nanosec
        if previous_clock is not None and value<previous_clock: clock_resets.append(record)
        previous_clock=value
    if previous_clock is None: raise ValueError('v4 requires recorded /clock')
    for i,(record,msg,ackrecord,ack) in enumerate(actions):
        if msg.producer_session_id not in sessions:sessions.append(msg.producer_session_id)
        anchor=min(record,ackrecord);picked=[asof(records.get(topic,[]),anchor) for topic in TOPICS]
        reasons=[]
        if events[i][1]:reasons.append(events[i][1])
        ages=np.array([record-p[0] if p else np.iinfo(np.int64).max for p in picked],dtype=np.int64)
        if any(p is None for p in picked):reasons.append('missing_observation')
        elif (ages>limits).any():reasons.append('stale_observation')
        elif max_camera_skew_s is not None and np.ptp([p[0] for p in picked[2:]])>max_camera_skew_s*1e9:reasons.append('camera_skew')
        if reasons:
            rejected.append(dict(index=i,record_ns=record,reasons=reasons));ticks.append(None);continue
        state,js,*images=[p[1] for p in picked]
        if state.header.frame_id!='world':raise ValueError('Invalid state frame')
        if names is None:names=list(js.name)
        if names!=list(js.name) or len(set(names))!=len(names):raise ValueError('Invalid/changing joint order')
        row={'state':flatten_state(state)}
        for field,key in [('position','joint_position'),('velocity','joint_velocity')]:
            a=np.asarray(getattr(js,field),dtype=np.float64)
            if a.shape!=(len(names),) or not np.isfinite(a).all():raise ValueError('Invalid measured joints')
            row[key]=a
        has_effort=bool(len(js.effort))
        if effort is not None and effort!=has_effort:raise ValueError('Changing effort availability')
        effort=has_effort
        if effort:row['joint_effort']=np.asarray(js.effort,dtype=np.float64)
        for c,image in zip(CAMERAS,images):
            row['images/'+c]=decode_image(image);encodings[c]=image.encoding
        row.update(source_record_ns=np.array([p[0] for p in picked],dtype=np.int64),
                   source_header_ns=np.array([header_ns(p[1]) for p in picked],dtype=np.int64),sync_age_ns=ages)
        ticks.append(row)
    clock_values={clock.clock.sec*10**9+clock.clock.nanosec for _,clock in records[CLOCK]}
    if any(header_ns(msg) not in clock_values or header_ns(ack) not in clock_values for _,msg,_,ack in actions):
        raise ValueError('Decision/acceptance stamps not witnessed by /clock')
    pairs=[]
    for i in range(len(actions)-1):
        a,m,ar,ack=actions[i];b,n,br,nack=actions[i+1]
        boundary=(events[i][0]!=events[i+1][0] or any(a<t<=b for t in clock_resets))
        if not boundary and m.producer_session_id!=n.producer_session_id:raise ValueError('Competing producer')
        if (ticks[i] is not None and ticks[i+1] is not None and not boundary
                and continuity_reason(header_ns(m),header_ns(n),timing) is None
                and continuity_reason(header_ns(ack),header_ns(nack),timing) is None):pairs.append(i)
        boundaries.append(boundary)
    if not pairs:raise ValueError('No causal accepted v4 transitions')
    # Verify command streams separately across actual reset/lifecycle boundaries.
    starts=[0]+[i+1 for i,b in enumerate(boundaries) if b];checks=[]
    for start,end in zip(starts,starts[1:]+[len(actions)]):
        subset=actions[start:end]
        begin=min(subset[0][0],subset[0][2]);finish=actions[end][0] if end<len(actions) else float('inf')
        streams={t:[s for s in samples if begin<=s[0]<finish] for t,samples in records.items()}
        checks.append(verify_streams(streams,subset,config))
    arrays={'action':np.stack([flatten_action(actions[i][1]) for i in pairs])}
    for prefix,shift in [('obs',0),('next_obs',1)]:
        for key in ticks[pairs[0]]:arrays[f'{prefix}/{key}']=np.stack([ticks[i+shift][key] for i in pairs])
    for pre,shift in [('',0),('next_',1)]:
        selected=[actions[i+shift] for i in pairs]
        arrays[pre+'record_ns']=np.array([r for r,_,_,_ in selected],dtype=np.int64)
        arrays[pre+'action_header_ns']=np.array([header_ns(m) for _,m,_,_ in selected],dtype=np.int64)
        arrays[pre+'acceptance_ns']=np.array([header_ns(ack) for _,_,_,ack in selected],dtype=np.int64)
        arrays[pre+'acceptance_record_ns']=np.array([r for _,_,r,_ in selected],dtype=np.int64)
        arrays[pre+'action_id']=np.array([m.action_id for _,m,_,_ in selected],dtype=np.uint64)
        arrays[pre+'producer_session_index']=np.array([sessions.index(m.producer_session_id) for _,m,_,_ in selected],dtype=np.int64)
    origin=int(arrays['acceptance_ns'][0]);endflags=np.r_[[(b!=a+1 or boundaries[a]) for a,b in zip(pairs,pairs[1:])],True]
    arrays.update(timestamp=(arrays['acceptance_ns']-origin)/1e9,next_timestamp=(arrays['next_acceptance_ns']-origin)/1e9,
                  duration=(arrays['next_acceptance_ns']-arrays['acceptance_ns'])/1e9,
                  terminated=np.zeros(len(pairs),dtype=bool),truncated=endflags,segment_id=np.cumsum(np.r_[0,endflags[:-1]]).astype(np.int64))
    attrs=dict(schema_name=SCHEMA,schema_version='4.0.0',profile=PROFILE,state_order=list(FIELDS),
               state_units={k:u for k,(_,u) in FIELDS.items()},contract=SharpaV4Contract().to_dict(),joint_names=names,
               joint_units=dict(position='rad',velocity='rad/s',effort='N*m'),has_effort=effort,sync_clock='mcap_log_time_ns',
               source_columns=list(SOURCES),max_age_ns=limits.tolist(),source_episode_id=bfm['episode']['episode_id'],
               collection_commit=collection_commit,source_metadata=metadata,success=metadata.get('success'),
               stored_image_encoding='rgb8',image_shapes={c:list(arrays['obs/images/'+c].shape[1:]) for c in CAMERAS},
               source_image_encodings=encodings,policy_timing=timing,policy_origin_ns=origin,record_origin_ns=actions[0][0],
               producer_sessions=sessions,execution=bfm['execution'],execution_checks=checks,
               total_action_ticks=len(actions),valid_decision_ticks=sum(t is not None for t in ticks),rejected_ticks=len(rejected),rejected_decisions=rejected)
    return arrays,attrs
