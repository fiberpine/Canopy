"""Exact native joint APPLY/no-op contract; legacy rows have implicit APPLY=1."""
def mapping_error(row):
 apply=row.get('apply','1')
 if apply not in ('0','1'):return True
 if row['state_id']!=row['action_id'] or int(row['return_code'])!=0:return True
 percent=int(row['percent']);before=int(row['cwnd_before'])
 if apply=='0':
  # native joint_cwnd_target returns the observed window unchanged; there is no
  # setsockopt and no minimum-window clamp on a no-op telemetry acknowledgment.
  return percent!=100 or int(row['cwnd_requested'])!=before
 expected=min(2147483647,max(4,percent*before//100))
 return int(row['cwnd_requested'])!=expected
