#ifndef _TIMER_H
#define _TIMER_H
void cpu_timer_start(struct timespec *tstart_cpu);
double cpu_timer_stop(struct timespec tstart_cpu);
#endif
