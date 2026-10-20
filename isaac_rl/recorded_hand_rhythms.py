"""Recorded Groove MIDI windows; hands are labeled by simulation, not inferred from MIDI."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from amortized_opt.rhythm import Hit,RhythmClip,quantize_time
from scripts.convert_groove_midi import parse_midi_hits,PITCH_TO_DRUM


class RecordedHandRhythmSampler:
    def __init__(self,kit,groups,eligibility,root,split='train'):
        if groups!=4:raise ValueError('Four-group ranker windows required')
        self.kit,self.groups,self.chart=kit,groups,eligibility
        self.sources=[];self.last_strata=None
        pitch_map={p:d for p,d in PITCH_TO_DRUM.items() if d!='kick' and p!=44}
        with (Path(root)/'info.csv').open() as stream:rows=list(csv.DictReader(stream))
        rejected=0
        for row in rows:
            if row['split']!=split:continue
            path=Path(root)/row['midi_filename']
            if not path.exists():continue
            hits,_,_,_=parse_midi_hits(path,pitch_map);buckets=defaultdict(list)
            for hit in hits:buckets[quantize_time(hit.time_sec)].append(hit)
            ordered=sorted(buckets.items());windows=[]
            for start in range(len(ordered)-groups+1):
                selected=ordered[start:start+groups]
                if any(len(h)>2 or len({x.drum for x in h})!=len(h) for _,h in selected):
                    rejected+=1;continue
                gaps=np.diff([t for t,_ in selected])
                if (gaps<.02-1e-7).any() or (gaps>.9+1e-7).any():rejected+=1;continue
                windows.append(start)
            if windows:self.sources.append(dict(path=str(path.resolve()),groups=ordered,windows=windows))
        if not self.sources:raise ValueError('No eligible recorded four-group windows')
        self.summary=dict(source='groove_midi_recorded_four_group_windows',groups=groups,
            source_split=split,performances=len(self.sources),windows=sum(len(s['windows']) for s in self.sources),
            rejected_windows=rejected,sampling='Uniform source performance then uniform eligible window',
            time_grid_s=.02,first_delay_s=[.02,.9],first_hit_policy='iid_random',
            velocity_source='Recorded MIDI velocity /127',human_hand_labels_available=False,
            labels='All legal robot assignments evaluated in Isaac',excluded_notes='Kick and pedal hi-hat44',
            citation='Gillick et al., Learning to Groove with Inverse Sequence Transformations, ICML2019',
            source_url='https://magenta.withgoogle.com/datasets/groove',license='CC BY4.0')

    def sample(self,count,rng,minimum_plan_choices=1):
        if minimum_plan_choices!=1:raise ValueError('Recorded collection includes forced legal plans')
        clips=[]
        for _ in range(count):
            source=self.sources[int(rng.integers(len(self.sources)))];start=int(rng.choice(source['windows']))
            selected=source['groups'][start:start+4];first=selected[0][0]
            lead=int(rng.integers(1,46))*.02;hits=[]
            for t,rows in selected:
                when=round(lead+t-first,8)
                for slot,row in enumerate(sorted(rows,key=lambda r:self.kit.names.index(r.drum))):
                    hits.append(Hit(when,when,self.kit.names.index(row.drum),row.drum,('left','right')[slot],row.velocity/127.))
            clips.append(RhythmClip(f'groove_{Path(source["path"]).stem}_window{start}_{int(rng.integers(2**63)):016x}',hits,hits[-1].time+.3))
        self.last_strata=np.zeros(count,dtype=np.int64)
        return clips

    def seed_payload(self):
        from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
        return StochasticHandRhythmSampler(self.kit,4,self.chart).seed_payload()
