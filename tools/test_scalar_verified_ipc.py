import importlib.util
import pathlib
import struct
import unittest
import zlib

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verified_ipc',str(ROOT/'frameworks/canopy_single/verified_ipc.py'))
ipc=importlib.util.module_from_spec(spec);spec.loader.exec_module(ipc)


class ProtocolTests(unittest.TestCase):
    def test_frame_round_trip(self):
        for kind in [ipc.STATE,ipc.ACTION,ipc.READY]:
            data=ipc.encode(kind,1,b'657 25 1 123')
            self.assertEqual(ipc.decode(data),(kind,1,zlib.crc32(b'657 25 1 123'),b'657 25 1 123'))
    def test_empty(self):
        self.assertIsNone(ipc.decode(b'\0'*2048))
    def test_header_corruption(self):
        data=bytearray(ipc.encode(ipc.STATE,1,b'657 1 2 3'))
        for i in range(32):
            changed=bytearray(data);changed[i]^=1
            with self.assertRaises(ipc.ProtocolError):ipc.decode(changed)
    def test_payload_corruption(self):
        data=bytearray(ipc.encode(ipc.STATE,1,b'657 1 2 3'));data[-1]^=1
        with self.assertRaises(ipc.ProtocolError):ipc.decode(data)
    def test_torn_copy(self):
        a=ipc.encode(ipc.STATE,1,b'657 '+b'1234 '*60)
        b=ipc.encode(ipc.STATE,2,b'658 '+b'56789 '*60)
        with self.assertRaises(ipc.ProtocolError):ipc.decode(b[:128]+a[128:])
    def test_boundaries_and_ready_separate(self):
        self.assertEqual(ipc.action_payload(0,25,1,0),b'0 25 1 0')
        self.assertEqual(ipc.action_payload(999,400,10,2**32-1),b'999 400 10 4294967295')
        for value in [24,401,99999,-1]:
            with self.assertRaises(ipc.ProtocolError):ipc.action_payload(657,value,1,123)
    def test_invalid_lengths(self):
        for payload in [b'',b'x'*2017,b'x\0y']:
            with self.assertRaises(ipc.ProtocolError):ipc.encode(ipc.STATE,1,payload)
    def test_invalid_sequence_and_kind(self):
        for kind,sequence in [(0,1),(4,1),(1,0),(1,-1),(1,2**64)]:
            with self.assertRaises(ipc.ProtocolError):ipc.encode(kind,sequence,b'123')
    def test_truncation(self):
        data=ipc.encode(ipc.STATE,1,b'657 1 2 3')
        for i in range(len(data)):
            with self.assertRaises(ipc.ProtocolError):ipc.decode(data[:i])


if __name__=='__main__':unittest.main()
