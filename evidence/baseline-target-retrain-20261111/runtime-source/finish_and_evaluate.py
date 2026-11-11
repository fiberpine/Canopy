"""Authorized phase transition, gated on the complete persisted training ledger."""
import contextlib
import io
import json
import time
import control

reg=control.registration();run='targetbn-base-seed3'
while time.time()<reg['train_cutoff_epoch']+120:
    with contextlib.redirect_stdout(io.StringIO()):rows=control.status(run)
    result=rows[0]['result']
    if result is not None:
        assert result['status']=='COMPLETED',result
        control.training_complete()
        control.export_final()
        control.launch_evaluation()
        control.save('phase-state.json',dict(status='FROZEN_EVALUATION_RUNNING',epoch=time.time()))
        break
    time.sleep(15)
else:
    raise RuntimeError('Training phase did not produce a complete persisted result')
