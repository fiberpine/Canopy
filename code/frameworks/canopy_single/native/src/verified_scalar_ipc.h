#ifndef CANOPY_VERIFIED_SCALAR_IPC_H
#define CANOPY_VERIFIED_SCALAR_IPC_H
#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

namespace scalar_ipc {
static const unsigned STATE=1, ACTION=2, READY=3, HEADER=32, SIZE=2048;
inline uint32_t crc32(const char* p,size_t n) {
    uint32_t c=0xffffffffU;
    for(size_t i=0;i<n;i++) { c^=(unsigned char)p[i]; for(int j=0;j<8;j++) c=(c>>1)^(0xedb88320U & (0U-(c&1U))); }
    return c^0xffffffffU;
}
inline void put(char* p,uint64_t v,int n) { for(int i=0;i<n;i++) p[i]=(char)((v>>(8*i))&255U); }
inline uint64_t get(const char* p,int n) { uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t((unsigned char)p[i])<<(8*i);return v; }
struct Frame { unsigned kind;uint64_t sequence;uint32_t crc;std::string payload; };
inline std::string encode(unsigned kind,uint64_t sequence,const std::string& payload) {
    if(kind<1 || kind>3 || !sequence || payload.empty() || payload.size()>SIZE-HEADER || payload.find('\0')!=std::string::npos)
        throw std::runtime_error("encode range");
    std::string out(HEADER,'\0');std::memcpy(&out[0],"ORCAIPC2",8);out[8]=2;out[9]=(char)kind;
    put(&out[12],payload.size(),4);put(&out[16],sequence,8);put(&out[24],crc32(payload.data(),payload.size()),4);
    put(&out[28],crc32(out.data(),28),4);out+=payload;return out;
}
inline bool decode(const char* data,size_t size,Frame& frame) {
    if(size<HEADER)throw std::runtime_error("short header");
    bool empty=true;for(unsigned i=0;i<HEADER;i++)if(data[i])empty=false;if(empty)return false;
    if(std::memcmp(data,"ORCAIPC2",8) || data[8]!=2 || data[9]<1 || data[9]>3 || get(data+10,2) || !get(data+16,8))
        throw std::runtime_error("schema/kind/sequence");
    if(get(data+28,4)!=crc32(data,28))throw std::runtime_error("header CRC");
    size_t n=get(data+12,4);if(!n || n>SIZE-HEADER || size<HEADER+n)throw std::runtime_error("payload length");
    if(get(data+24,4)!=crc32(data+HEADER,n))throw std::runtime_error("payload CRC");
    frame.kind=(unsigned char)data[9];frame.sequence=get(data+16,8);frame.crc=(uint32_t)get(data+24,4);frame.payload.assign(data+HEADER,n);
    if(frame.payload.find('\0')!=std::string::npos)throw std::runtime_error("payload NUL");return true;
}
class Channel {
    char* memory;int fd;uint64_t tx,rx;
    class Guard {
        int fd;
    public:
        explicit Guard(int value):fd(value) { while(flock(fd,LOCK_EX)<0)if(errno!=EINTR)throw std::runtime_error("flock"); }
        ~Guard(){flock(fd,LOCK_UN);}
    };
public:
    Channel(char* pointer,const std::string& path):memory(pointer),fd(-1),tx(0),rx(0) {
        fd=open(path.c_str(),O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("lock open");
    }
    ~Channel(){if(fd>=0)close(fd);}
    Frame write(unsigned kind,const std::string& payload) {
        std::string data=encode(kind,++tx,payload);
        { Guard lock(fd);std::memcpy(memory,data.data(),data.size()); }
        Frame frame;decode(data.data(),data.size(),frame);return frame;
    }
    bool read(Frame& frame) {
        char copy[SIZE];{ Guard lock(fd);std::memcpy(copy,memory,SIZE); }
        if(!decode(copy,SIZE,frame))return false;
        if(frame.sequence==rx)return false;
        if(frame.sequence!=rx+1)throw std::runtime_error("nonconsecutive sequence");
        rx=frame.sequence;return true;
    }
};
inline std::string bound_action(const Frame& f,int rid,const Frame& state) {
    if(f.kind!=ACTION || f.payload.find_first_not_of("0123456789 ")!=std::string::npos)throw std::runtime_error("action type/syntax");
    int id,pct;uint64_t sequence;uint32_t crc;std::string extra;std::istringstream in(f.payload);
    if(!(in>>id>>pct>>sequence>>crc) || (in>>extra) || id!=rid || id<0 || id>=1000 || pct<25 || pct>400 || sequence!=state.sequence || crc!=state.crc)
        throw std::runtime_error("action range/state binding");
    return std::to_string(id)+" "+std::to_string(pct);
}
}
#endif
