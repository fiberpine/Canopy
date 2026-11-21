"""Opt-in CRC framed, locked scalar IPC. Old ASCII transport remains the default.

Both peers must use this protocol. flock guards the complete shared-memory copy;
CRC and monotonic sequences additionally detect corruption/replay. No TCP reads.
"""
import contextlib
import csv
import hashlib
import json
import os
import pathlib
import struct
import zlib

MAGIC=b'ORCAIPC2'
HEADER=struct.Struct('<8sBBHIQII')
STATE,ACTION,READY=1,2,3
SIZE=2048


class ProtocolError(RuntimeError):
    pass


def encode(kind,sequence,payload):
    if kind not in (STATE,ACTION,READY) or not 0<sequence<2**64:
        raise ProtocolError('kind/sequence')
    if not isinstance(payload,bytes) or not 0<len(payload)<=SIZE-HEADER.size or b'\0' in payload:
        raise ProtocolError('payload length/NUL')
    crc=zlib.crc32(payload)&0xffffffff
    header=HEADER.pack(MAGIC,2,kind,0,len(payload),sequence,crc,0)
    return header[:28]+struct.pack('<I',zlib.crc32(header[:28])&0xffffffff)+payload


def decode(data):
    if len(data)<HEADER.size:
        raise ProtocolError('short header')
    if data[:HEADER.size]==b'\0'*HEADER.size:return None
    magic,version,kind,reserved,length,sequence,crc,hcrc=HEADER.unpack(data[:HEADER.size])
    if magic!=MAGIC or version!=2 or kind not in (STATE,ACTION,READY) or reserved or not sequence:
        raise ProtocolError('schema/kind/sequence')
    if hcrc!=(zlib.crc32(data[:28])&0xffffffff):raise ProtocolError('header CRC')
    if not 0<length<=SIZE-HEADER.size or len(data)<HEADER.size+length:
        raise ProtocolError('payload length')
    payload=data[HEADER.size:HEADER.size+length]
    if b'\0' in payload or (zlib.crc32(payload)&0xffffffff)!=crc:raise ProtocolError('payload CRC/NUL')
    return kind,sequence,crc,payload


def action_payload(rid,percent,state_sequence,state_crc):
    if not 0<=rid<1000 or not 24<=percent<=400 or state_sequence<=0 or not 0<=state_crc<2**32:
        raise ProtocolError('action range/state binding')
    return ('%d %d %d %d'%(rid,percent,state_sequence,state_crc)).encode('ascii')


def validate_native(path):
    path=pathlib.Path(path)
    metadata=json.loads(pathlib.Path(str(path)+'.ipc-schema.json').read_text())
    if metadata.get('schema')!=2 or metadata.get('binary_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ProtocolError('Explicit verified native/schema/SHA required')


class Channel:
    def __init__(self,shm,path):
        import fcntl
        self.fcntl=fcntl;self.shm=shm
        self.fd=os.open(str(path),os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
        self.tx=0;self.rx=0
        self.stats=dict(writes=0,received=0,duplicates=0,empty=0,rejected=0)

    @contextlib.contextmanager
    def locked(self):
        self.fcntl.flock(self.fd,self.fcntl.LOCK_EX)
        try:yield
        finally:self.fcntl.flock(self.fd,self.fcntl.LOCK_UN)

    def write(self,kind,payload):
        self.tx+=1;data=encode(kind,self.tx,payload)
        with self.locked():self.shm.write(data)
        self.stats['writes']+=1
        return decode(data)

    def read(self):
        with self.locked():data=self.shm.read()
        try:
            frame=decode(data)
            if frame is None:self.stats['empty']+=1;return None
            seq=frame[1]
            if seq==self.rx:self.stats['duplicates']+=1;return None
            if seq!=self.rx+1:raise ProtocolError('nonconsecutive sequence')
            self.rx=seq;self.stats['received']+=1;return frame
        except ProtocolError:
            self.stats['rejected']+=1;raise

    def close(self):
        os.close(self.fd)


class Pair:
    def __init__(self,read_shm,write_shm,folder):
        self.folder=pathlib.Path(folder);self.last_state=None;self.last_payload=b'\0'
        self.incoming=Channel(read_shm,self.folder/'ipc-state.lock')
        self.outgoing=Channel(write_shm,self.folder/'ipc-action.lock')
        self.file=(self.folder/'ipc-python.csv').open('w',buffering=65536)
        self.writer=csv.writer(self.file)
        self.writer.writerow(['direction','kind','sequence','crc','payload'])
        self.closed=False;pair=self
        class Reader:
            key=read_shm.key
            def read(self):
                frame=pair.incoming.read()
                if frame:
                    if frame[0]!=STATE:raise ProtocolError('Expected state frame')
                    pair.last_state=frame;pair.last_payload=frame[3]+b'\0';pair.log('received',frame)
                return pair.last_payload
            def remove(self):read_shm.remove()
        class Writer:
            key=write_shm.key
            def write(self,data):
                if isinstance(data,str):data=data.encode('ascii')
                values=data.split(b'\0')[0].split()
                if values==[b'99999',b'99999'] and pair.last_state is None:
                    frame=pair.outgoing.write(READY,b'99999 99999')
                else:
                    if len(values)!=2 or pair.last_state is None:raise ProtocolError('Unbound action')
                    rid,pct=map(int,values)
                    if rid!=int(pair.last_state[3].split()[0]):raise ProtocolError('State ID echo mismatch')
                    frame=pair.outgoing.write(ACTION,action_payload(rid,pct,pair.last_state[1],pair.last_state[2]))
                pair.log('sent',frame)
            def remove(self):write_shm.remove()
        self.reader,self.writer_shm=Reader(),Writer()

    def log(self,direction,frame):
        self.writer.writerow([direction,frame[0],frame[1],frame[2],frame[3].decode('ascii')])

    def close(self):
        if self.closed:return
        self.closed=True;self.file.close();self.incoming.close();self.outgoing.close()
        (self.folder/'ipc-python-stats.json').write_text(json.dumps(dict(schema=2,incoming=self.incoming.stats,
                outgoing=self.outgoing.stats,mechanism='Per-channel exclusive flock + CRC32 header/payload + monotonic sequence + state-bound action'),indent=2))
