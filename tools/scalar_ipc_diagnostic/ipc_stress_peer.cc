#include "verified_scalar_ipc.h"
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
char* mapping(const std::string& file) {
    int fd=open(file.c_str(),O_RDWR,0600);if(fd<0)throw std::runtime_error("map open");
    void* p=mmap(nullptr,2048,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
    if(p==MAP_FAILED)throw std::runtime_error("mmap");return (char*)p;
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)return 2;std::string root=argv[1];int count=std::atoi(argv[2]);
        scalar_ipc::Channel state(mapping(root+"/state.mem"),root+"/state.lock");
        scalar_ipc::Channel action(mapping(root+"/action.mem"),root+"/action.lock");
        for(int i=1;i<=count;i++) {
            std::string payload=std::to_string(i%1000)+" 1.25 "+std::to_string(i*17)+" 2 0.02 50 20 123456 0 20 2147483647 19 0 20 1448 20";
            auto s=state.write(scalar_ipc::STATE,payload);scalar_ipc::Frame a;
            while(!action.read(a))usleep(50);
            scalar_ipc::bound_action(a,i%1000,s);
        }
        std::printf("PASS %d cross-language state-bound exchanges\n",count);return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
