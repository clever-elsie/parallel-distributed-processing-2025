#ifndef HELPER_H
#define HELPER_H
#include <stdlib.h>
#include <omp.h>

int get_thread_num(){
  char* str=getenv("OMP_NUM_THREADS");
  if(str) return atoi(str);
  return omp_get_max_threads();
}

double duration_time(struct timeval const*begin, struct timeval const*end){
  return (double)(
    ((long)(end->tv_sec)-begin->tv_sec)*1000000
    +(end->tv_usec-begin->tv_usec)
  )/1000000;
}

#endif