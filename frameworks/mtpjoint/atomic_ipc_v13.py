"""Version-guarded snapshots for existing2048-byte SysV segments.

The ASCII payload/schema and state/action mappings are unchanged. Counter lives
outside the1000-byte native payload. Current Linux x86 hosts provide coherent
aligned64-bit loads; native publication uses compiler atomics/release fences.
"""
import struct
OFFSET=2032
def snapshot(shm):
 before=struct.unpack('=Q',shm.read(8,OFFSET))[0]
 if before&1:return None
 data=shm.read();after=struct.unpack('=Q',shm.read(8,OFFSET))[0]
 if before!=after or after&1 or len(data)<OFFSET+8:return None
 if struct.unpack('=Q',data[OFFSET:OFFSET+8])[0]!=after:return None
 return data
class Writer:
 def __init__(self,shm):self.shm=shm;self.sequence=0
 def write(self,data):
  if isinstance(data,str):data=data.encode('ascii')
  if len(data)>=OFFSET:raise ValueError('Atomic payload overlaps version counter')
  self.sequence+=1
  self.shm.write(struct.pack('=Q',2*self.sequence-1),OFFSET)
  self.shm.write(data)
  self.shm.write(struct.pack('=Q',2*self.sequence),OFFSET)
