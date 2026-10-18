#ifndef ORCA_JOINT_SAMPLE_CLOCK_H
#define ORCA_JOINT_SAMPLE_CLOCK_H
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

inline uint64_t joint_monotonic_us() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) abort();
    return uint64_t(ts.tv_sec)*1000000 + uint64_t(ts.tv_nsec)/1000;
}

// Consecutive observation endpoints partition physical time. Publication,
// inference, ACK waits and scheduling between endpoints must not disappear.
struct JointSampleClock {
    uint64_t previous;
    explicit JointSampleClock(uint64_t initial) : previous(initial) {}
    uint64_t observe(uint64_t endpoint) {
        if (endpoint <= previous) abort();
        uint64_t elapsed = endpoint - previous;
        previous = endpoint;
        return elapsed;
    }
};
#endif
