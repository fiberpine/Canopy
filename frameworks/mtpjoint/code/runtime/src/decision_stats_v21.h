#ifndef DECISION_STATS_V21_H
#define DECISION_STATS_V21_H
#include <cstdint>
struct DecisionStatsV21 {
 long double rtt_sum=0,rate_sum=0;uint64_t rtt_count=0,rate_count=0,loss=0;
 template<class T> void add(const T& x) {rtt_sum+=(long double)x.avg_urtt*x.cnt;rtt_count+=x.cnt;rate_sum+=(long double)x.thr*x.thr_cnt;rate_count+=x.thr_cnt;loss+=x.lost_bytes;}
 template<class T> void emit(T& x) {if(rtt_count)x.avg_urtt=rtt_sum/rtt_count;x.cnt=rtt_count;if(rate_count)x.thr=rate_sum/rate_count;x.thr_cnt=rate_count;x.lost_bytes=loss;*this=DecisionStatsV21();}
};
#endif
