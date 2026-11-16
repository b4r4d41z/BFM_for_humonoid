"""Decoupled camera freshness; recorder causality and simulation age stay separate."""
from bisect import bisect_right
import json
from types import SimpleNamespace

import numpy as np

from .mcap import CLOCK, EPISODE, RECORDING, lifecycle

CAMERA_CLOCK = 'Isaac ROS2CameraHelper simulation time (default useSystemTime=false)'
POLICY = dict(name='decoupled_camera_sim_time_v1', selection_clock='mcap_log_time_ns',
              camera_freshness_clock='action_header_simulation_ns', max_camera_age_ns=100_000_000,
              non_camera_freshness_clock='mcap_log_time_ns', epoch_evidence='recorded_clock_and_lifecycle_v1')


class CameraSync:
    def __init__(self, records, metadata):
        domains = metadata.get('clock_domains', {})
        camera = domains.get('camera_image.header.stamp') if isinstance(domains,dict) else None
        action = domains.get('/bfm/action.header.stamp') if isinstance(domains,dict) else None
        if (not isinstance(camera,str) or camera.split(';')[0] != CAMERA_CLOCK
                or not isinstance(action,str) or not action.startswith('selected /clock sample;')
                or metadata['bfm'].get('policy_clock', {}).get('domain') != 'isaac_simulation'):
            raise ValueError('Camera/Action simulation clock relationship is not declared')
        if any(type(m.clock.sec) is not int or m.clock.sec<0 or type(m.clock.nanosec) is not int
               or not 0<=m.clock.nanosec<10**9 for _,m in records[CLOCK]):
            raise ValueError('Invalid camera clock evidence')
        clocks = [(r, m.clock.sec*10**9+m.clock.nanosec) for r,m in records[CLOCK]]
        if not clocks or any(t<0 for _,t in clocks):
            raise ValueError('Missing/invalid camera clock evidence')
        self.resets = [r for (_,a),(r,b) in zip(clocks,clocks[1:]) if b<a]
        times = sorted({r for topic in (EPISODE,RECORDING) for r,_ in records.get(topic,[])})
        events = lifecycle(records,metadata['bfm']['episode']['episode_id'],times)
        previous = 0; barriers = list(self.resets)
        for record,(generation,_) in zip(times,events):
            if generation!=previous:barriers.append(record)
            previous = generation
        self.barriers = sorted(set(barriers)); self.ranges = {}; self.epoch_ranges = {}; self.stamps = {}
        for record,stamp in clocks:
            group = bisect_right(self.barriers,record)
            self.stamps.setdefault(group,set()).add(stamp)
            for key,table in ((group,self.ranges),(bisect_right(self.resets,record),self.epoch_ranges)):
                lo,hi = table.get(key,(stamp,stamp));table[key] = (min(lo,stamp),max(hi,stamp))

    def ages(self, record, stamp, source_records, source_stamps):
        """Return ages and rejection reason; never search for another image."""
        group = bisect_right(self.barriers,int(record))
        if stamp not in self.stamps.get(group,set()):
            return None,'unverified_action_clock_epoch'
        lo,hi = self.ranges[group]; epoch = bisect_right(self.resets,int(record))
        result = []
        for received,captured in zip(source_records,source_stamps):
            if received>record:return None,'future_camera_record'
            if captured<0:return None,'invalid_camera_timestamp'
            if bisect_right(self.barriers,int(received))!=group:return None,'camera_epoch_mismatch'
            if captured>stamp:return None,'future_camera_frame'
            if not lo<=captured<=hi:return None,'unverified_camera_clock_epoch'
            # A delayed pre-reset frame can otherwise masquerade as a fresh
            # post-reset frame when timestamps overlap. Headers carry no epoch ID.
            if any(a<=captured<=b for e,(a,b) in self.epoch_ranges.items() if e<epoch):
                return None,'ambiguous_camera_clock_epoch'
            age = int(stamp)-int(captured)
            if age>POLICY['max_camera_age_ns']:return None,'stale_camera_observation'
            result.append(age)
        return np.asarray(result,dtype=np.int64),None

    @classmethod
    def from_zarr(cls, root):
        """Use the existing hashed history for independent epoch validation."""
        records = {}
        for key,topic in (('clock',CLOCK),('episode',EPISODE),('recording',RECORDING)):
            g = root['control_history/'+key]; offsets = g['offsets'][:]; payload = g['payload'][:].tobytes()
            records[topic] = [(int(t),json.loads(payload[a:b],object_hook=lambda d:SimpleNamespace(**d)))
                              for t,a,b in zip(g['record_ns'][:],offsets[:-1],offsets[1:])]
        return cls(records,root.attrs['source_metadata'])
