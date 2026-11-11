"""Independent data-phase watchdog; no model session or inference calls."""
import pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import node_control as node
run=sys.argv[1];deadline=float(sys.argv[2])
while time.time()<deadline:
 status=node.status(run)
 if status['entries'] and all(not entry['alive'] for entry in status['entries']):sys.exit(0)
 time.sleep(5)
node.signal_stop(run,'AMENDED_DATA_PHASE_DEADLINE')
time.sleep(30)
node.cleanup(run)
