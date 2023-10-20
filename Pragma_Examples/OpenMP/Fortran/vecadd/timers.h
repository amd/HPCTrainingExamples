#ifndef _TIMER_H
#define _TIMER_H

#include <sys/time.h>

#ifdef __cplusplus
extern "C"
{
#endif

void cpu_timer_start(long long *sec, long long *usec);
double cpu_timer_stop(long long sec, long long usec);
void cpu_timer_accumulate(long long sec, long long usec, double *taccumulate);

#ifdef __cplusplus
}
#endif

#endif  /* _TIMER_H */

