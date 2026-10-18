"""New-experience update ledger; no wall-clock credit without inserted experience."""
import math

def limit(inserted,warmup,ratio,physical_seconds=0.,warmup_physical_seconds=None,
          basis='decision',per_base20ms=.08):
 if basis=='decision':return int(max(0,inserted-warmup)*ratio)
 if basis!='physical_exposure':raise ValueError('Unknown update credit basis')
 if not math.isfinite(physical_seconds) or physical_seconds<0 or not 0<per_base20ms<=1:
  raise ValueError('Invalid measured exposure/update rate')
 if inserted<warmup or warmup_physical_seconds is None:return 0
 if not math.isfinite(warmup_physical_seconds) or not 0<=warmup_physical_seconds<=physical_seconds:
  raise ValueError('Invalid recorded warmup exposure')
 return int(max(0,physical_seconds-warmup_physical_seconds)/.02*per_base20ms)
