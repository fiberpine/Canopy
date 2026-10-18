#ifndef ORCA_SCHEDULED_BOUNDARY_V30_H
#define ORCA_SCHEDULED_BOUNDARY_V30_H
#include <stdint.h>
inline bool scheduled_feedback_publishable(bool scheduled, bool initialized,
                                           uint32_t avg_rtt, uint32_t srtt_us,
                                           uint32_t min_rtt_us) {
    if (avg_rtt > 0) return true;
    // An empty RTT-sample interval is information, not an invalid TCP snapshot.
    // Retain first-RTT initialization and never invent a fresh RTT measurement.
    return scheduled && initialized && srtt_us > 0 && min_rtt_us > 0 &&
           min_rtt_us != UINT32_MAX;
}
#endif
