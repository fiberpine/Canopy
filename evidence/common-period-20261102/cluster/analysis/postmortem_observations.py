
import pathlib,json,csv,collections
R=pathlib.Path('/home/experiment/orca-common-period-cluster-20261102');assert (R/'evaluation/result.json').exists()
F=R/'A/framework';out=[]
for actor_path in (R/'A/host').glob('actor*.result.json'):
 actor=json.loads(actor_path.read_text());sums=collections.defaultdict(lambda:collections.defaultdict(float))
 for episode in actor['episodes']:
  d=F/'logs'/episode['run'];cfg=json.loads((d/'config.json').read_text())
  with (d/'interval-rate.csv').open() as f:rates={int(x['sequence']):x for x in csv.DictReader(f)}
  with (d/'decisions.csv').open() as f:decisions=list(csv.DictReader(f))
  for index,dec in enumerate(decisions):
   rate=rates[index+2];key=(episode['kind'],round(float(dec['planned_mtp_ms'])));a=sums[key]
   a['decisions']+=1;a['physical_seconds']+=float(rate['dt_us'])/1e6
   if int(rate['rtt_count'])==0:
    a['empty_rtt']+=1
    if float(rate['acked_bytes_per_s'])>0:a['empty_rtt_with_delivery']+=1
 for (kind,period),stats in sums.items():out.append(dict(actor=actor['actor'],kind=kind,period_ms=period,**dict(stats)))
(R/'A/host/offline-observation-diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
