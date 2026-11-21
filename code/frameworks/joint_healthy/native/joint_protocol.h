#ifndef CANOPY_HEALTHY_JOINT_PROTOCOL_H
#define CANOPY_HEALTHY_JOINT_PROTOCOL_H
#include <sstream>
#include <string>
#include <stdint.h>
#include <limits.h>
#include <stdexcept>

struct HealthyCommand {
    int id, percent, apply;
    unsigned wait_us;
    bool valid;
};
inline HealthyCommand parse_healthy_command(const char *text) {
    HealthyCommand out = {0, 100, 0, 20000, false};
    long long id, percent, apply, wait;
    std::string extra;
    std::istringstream input(text);
    if (!(input >> id >> percent >> apply >> wait) || (input >> extra)) return out;
    if (id < 0 || id >= 1000 || percent < 25 || percent > 400 ||
        (apply != 0 && apply != 1) || (apply == 0 && percent != 100) ||
        wait < 20000 || wait > 160000) return out;
    out.id = int(id); out.percent = int(percent); out.apply = int(apply);
    out.wait_us = unsigned(wait); out.valid = true; return out;
}
inline int healthy_cwnd_target(const HealthyCommand &command, uint32_t before) {
    if (!command.apply) return before > INT_MAX ? INT_MAX : int(before);
    uint64_t value = uint64_t(command.percent) * before / 100;
    return value > INT_MAX ? INT_MAX : (value < 4 ? 4 : int(value));
}
inline double healthy_interval_rate(uint64_t previous_bytes, uint64_t bytes,
                                    uint64_t previous_us, uint64_t now_us) {
    if (bytes < previous_bytes || now_us <= previous_us)
        throw std::runtime_error("nonmonotonic ACK counter/clock");
    return double(bytes - previous_bytes) * 1000000. / double(now_us - previous_us);
}
inline bool healthy_publishable(bool scheduled, bool initialized, unsigned rtt_count,
                                 unsigned average_rtt, unsigned srtt, unsigned min_rtt) {
    if (rtt_count > 0 && average_rtt > 0) return true;
    return scheduled && initialized && srtt > 0 && min_rtt > 0 && min_rtt != UINT_MAX;
}
#endif
