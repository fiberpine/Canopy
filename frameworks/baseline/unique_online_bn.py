"""Choose one fetchable Sonnet update wrapper per online moving statistic.

No new graph operation or ordering is introduced. The caller keeps its existing
joint actor/update fetch. Only the duplicate zero-width predictor writes go away.
"""


def _statistic_destinations(entry):
    pending=[entry.op if hasattr(entry,'op') else entry]
    seen=set();destinations=set()
    assignments=('Assign','AssignSub','AssignAdd','AssignVariableOp','AssignSubVariableOp','AssignAddVariableOp')
    references=('Identity','Switch','RefSwitch','Enter','RefEnter')
    while pending:
        op=pending.pop()
        if op in seen:continue
        seen.add(op)
        if op.type in assignments:
            variable=op.inputs[0].op
            while variable.type in references:variable=variable.inputs[0].op
            name=variable.name
            if name.startswith('abstract_actor/') and name.rsplit('/',1)[-1] in ('moving_mean','moving_variance'):
                destinations.add(name)
        pending.extend(t.op for t in op.inputs)
        pending.extend(op.control_inputs)
    return destinations


def select_online_updates(entries):
    grouped={}
    for entry in entries:
        destinations=_statistic_destinations(entry)
        if len(destinations)!=1:
            raise ValueError('Expected exactly one online moving-stat destination for '+entry.name)
        name=next(iter(destinations));grouped.setdefault(name,[]).append(entry)
    expected={'abstract_actor/abstract_actor/%s/%s'%(layer,stat)
              for layer in ('bn1','bn2') for stat in ('moving_mean','moving_variance')}
    if set(grouped)!=expected:raise ValueError('Unexpected online BN statistics: '+repr(sorted(grouped)))
    chosen=[];metadata=[]
    for destination in sorted(grouped):
        options=grouped[destination]
        canonical=[entry for entry in options if entry.name.startswith('abstract_actor/')]
        if len(canonical)!=1:raise ValueError('Expected one canonical Sonnet writer for '+destination)
        entry=canonical[0];chosen.append(entry)
        metadata.append(dict(destination=destination,selected=entry.name,discarded=[x.name for x in options if x is not entry]))
    return chosen,dict(mode='unique_online_BN_concurrent_with_actor',original_entries=len(entries),
        selected_entries=len(chosen),statistics=metadata,
        semantics='Original fetchable canonical Sonnet wrappers; one write per statistic; existing joint actor fetch retained. EMA multiplicity intentionally changes from duplicated writes to one.')
