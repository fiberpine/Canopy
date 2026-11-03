import pathlib,runpy
from common_period_deployment import flush_captures
try:
    runpy.run_path(str(pathlib.Path(__file__).with_name('common_period_frozen.py')),run_name='__main__')
finally:
    # evaluate.main's finally has already cleaned the network processes/IPC.
    flush_captures()
