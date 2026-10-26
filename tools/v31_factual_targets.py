#!/usr/bin/env python3
"""Physical-horizon factual targets from contiguous, frozen-policy TCP data.

This does not make off-policy n-step returns valid. Callers must independently
verify frozen deterministic behavior and keep training/evaluation data separate.
No extra observations, interpolation, or trace-capacity labels are introduced.
"""
import numpy as np


def factual_targets(data, horizon_s=20., train_fraction=.6):
    if horizon_s <= 0 or not 0 < train_fraction < 1:
        raise ValueError('Invalid physical horizon or split')
    n = len(data['reward'])
    if not all(len(data[k]) == n for k in
               ['s0', 's1', 'a', 'birth', 'actor_ids', 'duration', 'discount', 'terminal']):
        raise ValueError('Inconsistent trajectory lengths')
    duration = np.asarray(data['duration'], np.float64).reshape(-1)
    discount = np.asarray(data['discount'], np.float64).reshape(-1)
    reward = np.asarray(data['reward'], np.float64).reshape(-1)
    if not (np.isfinite(reward).all() and np.isfinite(duration).all() and
            np.isfinite(discount).all() and (duration > 0).all() and
            (discount > 0).all() and (discount <= 1).all()):
        raise ValueError('Nonfinite or invalid factual transition')
    result = []
    breaks = 0
    for actor in np.unique(data['actor_ids']):
        ix = np.where(data['actor_ids'] == actor)[0]
        ix = ix[np.argsort(data['birth'][ix])]
        segments = []
        start = 0
        for pos in range(1, len(ix)):
            before, after = ix[pos-1], ix[pos]
            if (np.asarray(data['terminal'][before]).any() or
                    not np.array_equal(data['s1'][before], data['s0'][after])):
                segments.append(ix[start:pos]); start = pos; breaks += 1
        segments.append(ix[start:])
        for segment, indices in enumerate(segments):
            elapsed = np.r_[0., np.cumsum(duration[indices])]
            split = elapsed[-1] * train_fraction
            # Endpoints must themselves be observed states/actions in this segment.
            for pos, index in enumerate(indices[:-1]):
                end = int(np.searchsorted(elapsed, elapsed[pos] + horizon_s))
                if end >= len(indices):
                    continue
                if elapsed[end] <= split:
                    partition = 'train'
                elif elapsed[pos] >= split + horizon_s:
                    partition = 'heldout'
                else:
                    continue
                window = indices[pos:end]
                weights = np.r_[1., np.cumprod(discount[window])]
                result.append(dict(index=int(index), one_endpoint=int(indices[pos+1]),
                    endpoint=int(indices[end]), actor=int(actor), segment=segment,
                    partition=partition, start_s=float(elapsed[pos]), end_s=float(elapsed[end]),
                    horizon_s=float(elapsed[end]-elapsed[pos]),
                    observed_return=float(np.dot(weights[:-1], reward[window])),
                    tail_discount=float(weights[-1])))
    return result, dict(rows=n, targets=len(result), discontinuities=breaks,
        horizon_min_s=horizon_s, split='first60% train; heldout starts20s after split'
        if train_fraction == .6 and horizon_s == 20 else
        'train endpoints before split; heldout starts one horizon after split',
        labels='Factual frozen-policy returns, not optimal or counterfactual Q')
