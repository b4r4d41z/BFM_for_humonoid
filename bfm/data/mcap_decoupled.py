"""Causal 15Hz passive samples with full-rate history, never fabricated ACKs."""
import numpy as np
from bfm.contracts.sharpa import (SCHEMA,PROFILE,FIELDS,GROUPS,MODEL_SHA256,SharpaDecoupledContract,
                                  DECOUPLED_SCHEMA,decoupled_control,flatten_action,flatten_state)
from .mcap import ACTION,TOPICS,CLOCK,EPISODE,RECORDING,lifecycle,asof,header_ns,decode_image
from .decoupled import verify_commands,causal_target,history_arrays,HISTORY,arm_values,hand_values,ARMS,HANDS
from .policy_time import policy_timing,continuity_reason
from .sharpa import CAMERAS,SOURCES
from .execution_v4 import identity


def build(records,metadata,*,max_age_s,collection_commit,max_camera_skew_s=None):
    control=decoupled_control(metadata);bfm=metadata['bfm']
    if records.get('/bfm/execution_status'):raise ValueError('Executor evidence conflicts with decoupled recording architecture')
    if (metadata.get('robot')!='kuavo_s40_sharpa' or metadata.get('camera_configuration',{}).get('model_sha256')!=MODEL_SHA256
            or metadata.get('recording_code',{}).get('commit_sha')!=collection_commit or len(collection_commit)!=40
            or any(c not in '0123456789abcdef' for c in collection_commit)
            or metadata.get('time',{}).get('synchronization_source')!='rosbag_record_timestamp'):
        raise ValueError('Decoupled robot/model/collection/clock provenance mismatch')
    for samples in records.values():samples.sort(key=lambda x:x[0])
    command_warnings=verify_commands(records)
    actions=records[ACTION];timing=policy_timing(bfm,records[CLOCK])
    limits=np.asarray(max_age_s,dtype=float)
    if limits.shape!=(5,) or not np.isfinite(limits).all() or (limits<=0).any():raise ValueError('Five positive freshness limits required')
    limits=np.rint(limits*1e9).astype(np.int64)
    if max_camera_skew_s is not None and (not np.isfinite(max_camera_skew_s) or max_camera_skew_s<=0):raise ValueError('Invalid camera skew')
    events=lifecycle(records,bfm['episode']['episode_id'],[t for t,_ in actions])
    resets=[];previous=None;clock_values=set();clock_gaps=[];step=1e9/control['physics_hz']
    for record,msg in records[CLOCK]:
        stamp=msg.clock.sec*10**9+msg.clock.nanosec;clock_values.add(stamp)
        if previous is not None:
            delta=stamp-previous
            if delta<0:resets.append(record)
            elif delta>step+2:clock_gaps.append(record)
        previous=stamp
    import json
    cutoffs=list(resets);last_boundary=None
    for record,msg in records.get(EPISODE,[]):
        event=json.loads(msg.data)
        if 'phase' not in event:continue  # Recording events do not reset the passive sampler cache.
        boundary=event.get('boundary_id')
        if boundary!=last_boundary or event['phase'] in ('stopped','failed','closed'):cutoffs.append(record)
        last_boundary=boundary
    sessions=[];watermarks={};ticks=[];rejects=[];boundaries=[];names=None;effort=None;encodings={}
    lower_bound=-1
    for i,(record,msg) in enumerate(actions):
        key=identity(msg);a=flatten_action(msg)
        if any(np.asarray(getattr(msg,g+'_target')).shape!=(s.stop-s.start,) for g,s in GROUPS.items()):raise ValueError('Invalid action group width')
        if key[1]<=watermarks.get(key[0],0):raise ValueError('Duplicate/backward sampled action identity')
        watermarks[key[0]]=key[1]
        if key[0] not in sessions:sessions.append(key[0])
        stamp=header_ns(msg)
        if stamp<0 or stamp not in clock_values:raise ValueError('Action stamp not witnessed by recorded /clock')
        boundary=i>0 and (events[i][0]!=events[i-1][0] or any(actions[i-1][0]<r<=record for r in resets))
        lower_bound=max((r for r in cutoffs if r<=record),default=-1)
        boundaries.append(boundary)
        if i and not boundary and stamp<=header_ns(actions[i-1][1]):raise ValueError('Duplicate/backward decision stamp without reset')
        if i and not boundary and key[0]!=actions[i-1][1].producer_session_id:raise ValueError('Competing passive sampler session')
        provenance=causal_target(records,record,a,lower_bound)
        picked=[asof(records.get(topic,[]),record) for topic in TOPICS];reasons=[]
        if provenance is None:reasons.append('incomplete_boundary_command_provenance')
        if events[i][1]:reasons.append(events[i][1])
        ages=np.array([record-p[0] if p else np.iinfo(np.int64).max for p in picked],dtype=np.int64)
        if any(p is None for p in picked):reasons.append('missing_observation')
        elif any(p[0]<lower_bound for p in picked):reasons.append('pre_boundary_observation')
        elif (ages>limits).any():reasons.append('stale_observation')
        elif max_camera_skew_s is not None and np.ptp([p[0] for p in picked[2:]])>max_camera_skew_s*1e9:reasons.append('camera_skew')
        if reasons:
            rejects.append(dict(index=i,record_ns=record,action_header_ns=stamp,reasons=reasons));ticks.append(None);continue
        state,js,*images=[p[1] for p in picked]
        if state.header.frame_id!='world':raise ValueError('Expected world task state')
        if names is None:names=list(js.name)
        if names!=list(js.name) or len(set(names))!=len(names):raise ValueError('Changing/duplicate measured joints')
        row=dict(state=flatten_state(state),source_record_ns=np.array([p[0] for p in picked],dtype=np.int64),
                 source_header_ns=np.array([header_ns(p[1]) for p in picked],dtype=np.int64),sync_age_ns=ages)
        for field in ('position','velocity'):
            a=np.asarray(getattr(js,field),dtype=np.float64)
            if a.shape!=(len(names),) or not np.isfinite(a).all():raise ValueError('Invalid measured joints')
            row['joint_'+field]=a
        has=bool(len(js.effort))
        if effort is not None and effort!=has:raise ValueError('Changing effort availability')
        effort=has
        if effort:row['joint_effort']=np.asarray(js.effort,dtype=np.float64)
        from bfm.contracts.vision import proprioception
        proprioception(names,row['joint_position'],row['joint_velocity'])
        for camera,image in zip(CAMERAS,images):row['images/'+camera]=decode_image(image);encodings[camera]=image.encoding
        ticks.append((row,provenance))
    pairs=[];rejected_intervals=[]
    for i in range(len(actions)-1):
        begin,m=actions[i];end,n=actions[i+1];reason=continuity_reason(header_ns(m),header_ns(n),timing)
        if boundaries[i+1]:reason='episode_or_simulation_reset'
        elif any(begin<r<=end for r in clock_gaps):reason='missing_physics_clock_evidence'
        elif ticks[i] is None or ticks[i+1] is None:reason='invalid_endpoint'
        if reason:rejected_intervals.append(dict(index=i,reason=reason))
        else:pairs.append(i)
    if not pairs:raise ValueError('No verifiable decoupled transitions')
    arrays={'action':np.stack([flatten_action(actions[i][1]) for i in pairs])}
    for prefix,shift in [('obs',0),('next_obs',1)]:
        for name in ticks[pairs[0]][0]:arrays[prefix+'/'+name]=np.stack([ticks[i+shift][0][name] for i in pairs])
    for prefix,shift in [('',0),('next_',1)]:
        selected=[actions[i+shift] for i in pairs]
        arrays[prefix+'record_ns']=np.array([r for r,_ in selected],dtype=np.int64)
        arrays[prefix+'action_header_ns']=np.array([header_ns(m) for _,m in selected],dtype=np.int64)
        arrays[prefix+'action_id']=np.array([m.action_id for _,m in selected],dtype=np.uint64)
        arrays[prefix+'producer_session_index']=np.array([sessions.index(m.producer_session_id) for _,m in selected],dtype=np.int64)
        arrays[prefix+'command_source_record_ns']=np.stack([ticks[i+shift][1] for i in pairs])
    origin=int(arrays['action_header_ns'][0]);ends=np.r_[[b!=a+1 for a,b in zip(pairs,pairs[1:])],True]
    arrays.update(timestamp=(arrays['action_header_ns']-origin)/1e9,next_timestamp=(arrays['next_action_header_ns']-origin)/1e9,
                  duration=(arrays['next_action_header_ns']-arrays['action_header_ns'])/1e9,
                  terminated=np.zeros(len(pairs),dtype=bool),truncated=ends,segment_id=np.cumsum(np.r_[0,ends[:-1]]).astype(np.int64))
    history,history_meta=history_arrays(records,arrays['record_ns'],arrays['next_record_ns']);arrays.update(history)
    for key,indices,topic,extract in [('arm',ARMS,HISTORY['arm'],arm_values),('hand',HANDS,HISTORY['hand'],hand_values)]:
        delta=[];boundary_delta=[]
        for action,begin,end in zip(arrays['action'],arrays['record_ns'],arrays['next_record_ns']):
            initial=extract(asof(records[topic],begin)[1])-action[indices];boundary_delta.append(initial)
            values=[np.abs(initial)]+[np.abs(extract(m)-action[indices]) for r,m in records[topic] if begin<r<=end]
            delta.append(np.max(values,axis=0) if values else np.zeros(len(indices)))
        arrays['intervening_'+key+'_max_delta']=np.asarray(delta,dtype=np.float64)
        arrays['boundary_controller_'+key+'_delta']=np.asarray(boundary_delta,dtype=np.float64)
    attrs=dict(schema_name=SCHEMA,schema_version=DECOUPLED_SCHEMA,profile=PROFILE,state_order=list(FIELDS),
               state_units={k:u for k,(_,u) in FIELDS.items()},contract=SharpaDecoupledContract().to_dict(),
               control_architecture=control['control_architecture'],control=control,control_history=history_meta,
               boundary_command_warnings=command_warnings,
               stream_statistics={topic:dict(count=len(samples),first_record_ns=samples[0][0],last_record_ns=samples[-1][0],record_delta_ns_p50_p95_max=np.percentile(np.diff(np.array([r for r,_ in samples],dtype=np.int64)),[50,95,100]).tolist() if len(samples)>1 else None) for topic,samples in records.items() if samples},
               camera_skew_ns_p50_p95_max=np.percentile([np.ptp(t[0]['source_record_ns'][2:]) for t in ticks if t is not None],[50,95,100]).tolist(),
               source_age_ns_p50_p95_max=np.percentile(np.stack([t[0]['sync_age_ns'] for t in ticks if t is not None]),[50,95,100],axis=0).tolist(),
               sampled_control_approximation=True,command_source_columns=['mapped_arm','arm','raw_hand','hand'],
               provenance_limitations=['raw bridge to mapped arm unit/side overrides are not recorded; mapped radian source is authoritative',
                                       'no ACK or measured controller application time; snapshots do not describe all intervening inputs'],
               joint_names=names,joint_units=dict(position='rad',velocity='rad/s',effort='N*m'),has_effort=effort,
               sync_clock='mcap_log_time_ns',source_columns=list(SOURCES),max_age_ns=limits.tolist(),
               source_episode_id=bfm['episode']['episode_id'],collection_commit=collection_commit,source_metadata=metadata,
               success=metadata.get('success'),stored_image_encoding='rgb8',source_image_encodings=encodings,
               image_shapes={c:list(arrays['obs/images/'+c].shape[1:]) for c in CAMERAS},policy_timing=timing,
               policy_origin_ns=origin,record_origin_ns=actions[0][0],producer_sessions=sessions,
               total_action_ticks=len(actions),valid_decision_ticks=sum(t is not None for t in ticks),rejected_ticks=len(rejects),
               rejected_decisions=rejects,rejected_intervals=rejected_intervals,
               max_camera_skew_ns=int(max_camera_skew_s*1e9) if max_camera_skew_s is not None else None)
    return arrays,attrs
