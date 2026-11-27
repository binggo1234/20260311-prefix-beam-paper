"""Private bounded-window policies; no geometry or solver dependencies."""
from collections import OrderedDict
import hashlib
import random


class LazyPermutation:
    def __init__(self,size,seed):
        if type(size) is not int or size<0:raise ValueError('nonnegative size required')
        self.remaining=size
        self.rng=random.Random(seed)
        self.swaps={}

    def __iter__(self):return self

    def __next__(self):
        if not self.remaining:raise StopIteration
        index=self.rng.randrange(self.remaining)
        value=self.swaps.get(index,index)
        last=self.remaining-1
        self.swaps[index]=self.swaps.get(last,last)
        self.swaps.pop(last,None)
        self.remaining=last
        return value


def prefix_key(h,prefix):
    prefix=tuple(prefix)
    if not prefix:return tuple(h[:2])
    if len(prefix)==1:
        if len(h)==1:return prefix
        return (prefix[0],h[0] if h[0]!=prefix[0] else h[1])
    if len(prefix)==2 and prefix[0]!=prefix[1]:return prefix
    raise ValueError('distinct prefix of length at most two required')


def family_key(h,sequence):
    sequence=tuple(sequence)
    if len(sequence)!=len(h) or len(set(sequence))!=len(h) or set(sequence)!=set(h):return None
    key=sequence[:2]
    return key if sequence==key+tuple(u for u in h if u not in key) else None


def direct_pairs(h,seed):
    ids=tuple(sorted(h));n=len(ids)
    if n<2:return
    for index in LazyPermutation(n*(n-1),seed):
        u=index//(n-1);v=index%(n-1)
        if v>=u:v+=1
        yield ids[u],ids[v]


def tie_key(seed,u,v):
    return hashlib.blake2b(f'{seed}:{u}:{v}'.encode(),digest_size=8).digest()


def overlap(a,b):
    return sum(w*h*min(count,b.get((w,h),0)) for (w,h),count in a.items())


class StreamPolicy:
    def __init__(self,h,arm,seed,known,window_size=16,pairs_per_probe=2):
        self.h=tuple(h);self.ids=tuple(sorted(h))
        if not self.h or len(set(h))!=len(h):raise ValueError('unique nonempty identities required')
        if arm not in ('STREAM_OVERLAP','STREAM_RANDOM'):raise ValueError('unknown arm')
        if window_size<2 or pairs_per_probe<1:raise ValueError('positive policy bounds required')
        self.arm=arm;self.seed=seed;self.window_size=window_size;self.pairs_per_probe=pairs_per_probe
        self.probe_order=LazyPermutation(len(self.ids),seed)
        self.window=OrderedDict();self.single_cache={};self.seen=set(known)
        self.pending=None;self.slots=0;self.draining=False
        self.max_candidates_checked=0;self.max_window=0

    def _add_window(self,uid,entry):
        self.window[uid]=entry
        while len(self.window)>self.window_size:self.window.popitem(last=False)
        self.max_window=max(self.max_window,len(self.window))
        self.slots=self.pairs_per_probe

    def _best_pair(self):
        checked=0;best=None;best_key=None
        for u,a in self.window.items():
            for v,b in self.window.items():
                if u==v:continue
                checked+=1
                if (u,v) in self.seen:continue
                value=overlap(a['counts'],b['counts']) if self.arm=='STREAM_OVERLAP' else 0
                priority=(value,tie_key(self.seed,u,v),u,v)
                if best_key is None or priority<best_key:best_key=priority;best=(u,v)
        self.max_candidates_checked=max(self.max_candidates_checked,checked)
        return best

    def next_action(self):
        if self.pending is not None:raise RuntimeError('observe pending evaluation first')
        while True:
            if self.slots or self.draining:
                pair=self._best_pair()
                if pair is not None:
                    if self.slots:self.slots-=1
                    self.pending={'phase':'pair','prefix':pair}
                    return self.pending.copy()
                self.slots=0
                if self.draining:raise StopIteration
            try:uid=self.ids[next(self.probe_order)]
            except StopIteration:
                self.draining=True
                continue
            cached=self.single_cache.get(uid)
            if cached is not None:
                self._add_window(uid,cached)
                return {'phase':'feedback_reuse','prefix':(uid,),
                        'source_evaluation':cached['source_evaluation']}
            self.pending={'phase':'single_probe','prefix':(uid,)}
            return self.pending.copy()

    def record_target(self,prefix,counts,area,evaluation):
        key=prefix_key(self.h,prefix)
        self.seen.add(key)
        uid=key[0]
        if key==prefix_key(self.h,(uid,)):
            self.single_cache[uid]={'counts':counts.copy(),'area':area,'source_evaluation':evaluation}

    def observe(self,counts,area,evaluation):
        if self.pending is None:raise RuntimeError('no pending evaluation')
        action=self.pending
        self.record_target(action['prefix'],counts,area,evaluation)
        if action['phase']=='single_probe':
            uid=action['prefix'][0]
            self._add_window(uid,self.single_cache[uid])
        self.pending=None
