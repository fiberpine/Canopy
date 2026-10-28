
#include "verified_scalar_ipc.h"
#include <iostream>
int main(){using namespace scalar_ipc;Frame s,f;std::string raw=encode(STATE,1,"657 state");decode(raw.data(),raw.size(),s);raw=encode(ACTION,1,std::string("657 24 1 ")+std::to_string(s.crc));decode(raw.data(),raw.size(),f);if(bound_action(f,657,s)!="657 24")return 2;int rejected=0;f.payload=std::string("657 23 1 ")+std::to_string(s.crc);try{bound_action(f,657,s);}catch(...){rejected++;}f.payload="657 24 2 1";try{bound_action(f,657,s);}catch(...){rejected++;}if(rejected!=2)return 3;std::cout<<"PASS_24_ACCEPT_23_AND_STALE_REJECT";}
