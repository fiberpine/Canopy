"""Real Linux Python/C++ mmap IPC stress; no TF or TCP, no synthetic-network claim."""
import json
import os
import mmap
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT=pathlib.Path(os.environ['CANOPY_IPC_DIAGNOSTIC_ROOT']).resolve()
sys.path.insert(0,str(ROOT/'frameworks/canopy_single'))
from verified_ipc import Channel,STATE,ACTION,action_payload


class Memory:
    def __init__(self,path):
        with path.open('wb') as f:f.truncate(2048)
        self.f=path.open('r+b');self.map=mmap.mmap(self.f.fileno(),2048)
    def read(self):return self.map[:]
    def write(self,data):self.map[:len(data)]=data
    def close(self):self.map.close();self.f.close()


with tempfile.TemporaryDirectory(prefix='ipc-stress-',dir=str(ROOT)) as directory:
    folder=pathlib.Path(directory);sr=Memory(folder/'state.mem');aw=Memory(folder/'action.mem')
    read=Channel(sr,folder/'state.lock');write=Channel(aw,folder/'action.lock')
    count=5000;deadline=time.monotonic()+60
    child=subprocess.Popen([str(ROOT/'native/ipc-stress-peer'),str(folder),str(count)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(1,count+1):
            while True:
                f=read.read()
                if f:break
                if time.monotonic()>deadline:raise RuntimeError('IPC stress timeout')
                time.sleep(.00005)
            assert f[0]==STATE and f[1]==i
            write.write(ACTION,action_payload(int(f[3].split()[0]),25+(i%376),f[1],f[2]))
        out,err=child.communicate(timeout=10);assert child.returncode==0,(out,err)
        result=dict(status='PASS_REAL_LINUX_CROSS_LANGUAGE_IPC',exchanges=count,read=read.stats,write=write.stats,stdout=out.decode(),tcp_runs=0)
        (ROOT/'ipc-stress-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        if child.poll() is None:child.kill();child.wait()
        read.close();write.close();sr.close();aw.close()
