#include "joint_protocol.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    HealthyCommand a=parse_healthy_command("999 100 1 20000");
    assert(a.valid && a.id==999 && a.wait_us==20000 && healthy_cwnd_target(a,10)==10);
    assert(!parse_healthy_command("1000 100 1 20000").valid);
    assert(!parse_healthy_command("0 100 1 -1").valid);
    assert(!parse_healthy_command("0 100 1 160001").valid);
    assert(!parse_healthy_command("0 100 2 20000").valid);
    assert(!parse_healthy_command("0 99 0 20000").valid);
    assert(!parse_healthy_command("0 100 1 20000 extra").valid);
    assert(!parse_healthy_command("0 100.5 1 20000").valid);
    assert(!parse_healthy_command("99999 99999").valid);
    a=parse_healthy_command("0 400 1 160000");assert(a.valid && healthy_cwnd_target(a,UINT_MAX)==INT_MAX);
    a=parse_healthy_command("0 25 1 20000");assert(healthy_cwnd_target(a,4)==4);
    a=parse_healthy_command("0 100 0 20000");assert(a.valid && healthy_cwnd_target(a,3)==3);
    assert(healthy_interval_rate(1000,2000,100,20100)==50000.);
    assert(healthy_interval_rate(1000,1000,100,20100)==0.);
    bool rejected=false;try{healthy_interval_rate(1001,1000,100,20100);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
    assert(!healthy_publishable(true,false,0,0,20000,20000));
    assert(!healthy_publishable(false,true,0,0,20000,20000));
    assert(healthy_publishable(true,true,0,0,20000,20000));
    assert(healthy_publishable(false,false,1,20000,20000,20000));
    for(int percent=25;percent<=400;++percent) {
        a.percent=percent;a.apply=1;
        assert(healthy_cwnd_target(a,100)==percent);
    }
    std::cout<<"PASS_STRICT_PROTOCOL_RATE_FRESHNESS_AND_INTEGER_CWND\n";
}
