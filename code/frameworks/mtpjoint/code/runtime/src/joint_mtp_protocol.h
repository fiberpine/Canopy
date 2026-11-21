#ifndef JOINT_MTP_PROTOCOL_H
#define JOINT_MTP_PROTOCOL_H
#include <sstream>
#include <string>
#include <climits>
#include <cstdint>
#include <sys/socket.h>

struct JointCommand { int id, percent, apply; bool valid; unsigned int wait_us; bool has_wait; };
inline JointCommand parse_joint_command(const char *text) {
    JointCommand c = {0, 0, 0, false, 20000, false};
    std::istringstream input(text);
    std::string extra;
    if (!(input >> c.id >> c.percent >> c.apply)) return c;
    if(input >> extra) {
        std::istringstream duration(extra); uint64_t us; std::string bad;
        if(!(duration >> us) || (duration >> bad) || us<20000 || us>160000) return c;
        c.wait_us=static_cast<unsigned int>(us);c.has_wait=true;
        if(input >> extra) return c;
    }
    c.valid = c.id >= 0 && c.id < 1000 && c.percent >= 25 && c.percent <= 400
        && (c.apply == 1 || (c.apply == 0 && c.percent == 100));
    return c;
}
inline int joint_cwnd_target(const JointCommand &c, uint32_t cwnd) {
    if (!c.apply) return static_cast<int>(cwnd);
    uint64_t target = static_cast<uint64_t>(c.percent) * cwnd / 100;
    return target > INT_MAX ? INT_MAX : (target < 4 ? 4 : static_cast<int>(target));
}
typedef int (*JointSetSockOpt)(int, int, int, const void *, socklen_t);
inline int execute_joint_cwnd(const JointCommand &c, int socket, int level,
                             int option, int target, JointSetSockOpt setter) {
    return c.apply ? setter(socket, level, option, &target, sizeof(target)) : 0;
}
#endif
