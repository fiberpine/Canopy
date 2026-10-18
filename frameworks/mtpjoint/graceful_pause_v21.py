"""File-based pause without terminating the in-process training coordinator."""
import json,os,pathlib

def pause_request(root,cfg=None):
    candidates=[]
    if os.environ.get('ORCA_PAUSE_FILE'):candidates.append(pathlib.Path(os.environ['ORCA_PAUSE_FILE']))
    if cfg and cfg.get('stop_file'):candidates.append(pathlib.Path(cfg['stop_file']))
    for path in candidates:
        if path.exists():
            try:value=json.loads(path.read_text())
            except (ValueError,OSError):value={}
            return dict(reason=value.get('reason') or 'USER_REQUESTED_SAFE_PAUSE',path=str(path))
    return None

def may_start(root):
    return pause_request(root) is None
