#ifndef ORCA_INTERVAL_RATE_V28_H
#define ORCA_INTERVAL_RATE_V28_H
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include "joint_sample_clock.h"
// Linux v4.13 TCP_INFO UAPI: bytes_acked at offset120, uint64.
// Runtime returned length is checked; never read an unsupported extension.
struct IntervalRateV28 {
    FILE *log;
    bool replace;
    uint64_t previous_bytes, previous_mono;
    IntervalRateV28():log(NULL),replace(false),previous_bytes(0),previous_mono(0) {}
    uint64_t read(int fd) {
        unsigned char bytes[256]; memset(bytes,0,sizeof(bytes)); socklen_t n=sizeof(bytes);
        if(getsockopt(fd,IPPROTO_TCP,11,bytes,&n)!=0 || n<128) {
            fprintf(stderr,"V28_TCP_INFO_ABI_REQUIRED length=%u\n",unsigned(n)); abort();
        }
        uint64_t value; memcpy(&value,bytes+120,8);return value;
    }
    void begin(int fd) {
        const char *mode=getenv("ORCA_INTERVAL_RATE_V28");
        if(!mode) return;
        if(strcmp(mode,"audit_only")!=0 && strcmp(mode,"acked_bytes")!=0) abort();
        replace=strcmp(mode,"acked_bytes")==0;
        log=fopen(getenv("ORCA_INTERVAL_RATE_LOG"),"w");if(!log)abort();
        setvbuf(log,NULL,_IOFBF,65536);
        fprintf(log,"sequence,previous_mono_us,sample_mono_us,previous_bytes_acked,bytes_acked,delta_bytes,dt_us,acked_bytes_per_s,orca_delivery_sample_mean,lost_bytes,rtt_count,rate_count\n");
        previous_bytes=read(fd);previous_mono=joint_monotonic_us();
    }
    double sample(int fd,uint64_t sequence,uint64_t &mono,double old_rate,
                  unsigned loss,unsigned rtt_count,unsigned rate_count) {
        if(!log)return old_rate;
        uint64_t now_bytes=read(fd);mono=joint_monotonic_us();
        if(now_bytes<previous_bytes || mono<=previous_mono)abort();
        uint64_t delta=now_bytes-previous_bytes,dt=mono-previous_mono;
        double rate=double(delta)*1000000./double(dt);
        fprintf(log,"%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.12g,%.12g,%u,%u,%u\n",
            (unsigned long long)sequence,(unsigned long long)previous_mono,
            (unsigned long long)mono,(unsigned long long)previous_bytes,
            (unsigned long long)now_bytes,(unsigned long long)delta,
            (unsigned long long)dt,rate,old_rate,loss,rtt_count,rate_count);
        previous_bytes=now_bytes;previous_mono=mono;
        return replace?rate:old_rate;
    }
};
#endif
