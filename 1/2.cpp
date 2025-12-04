#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <mutex>
#include <atomic>
#include "0.h"

typedef struct{
  double time;
  const char* name;
} pair;

const int limit=1000000;

pair seq(){
  int sum=0;
  struct timeval begin,end;
  gettimeofday(&begin,NULL);
  for(int i=0;i<limit;++i)
    ++sum;
  gettimeofday(&end,NULL);
  return (pair){duration_time(&begin,&end), __func__};
}

pair cri(){
  int sum=0;
  struct timeval begin,end;
  gettimeofday(&begin,NULL);
#  pragma omp parallel for
  for(int i=0;i<limit;++i){
#   pragma omp critical
    ++sum;
  }
  gettimeofday(&end,NULL);
  return (pair){duration_time(&begin,&end), __func__};
}

pair cri_mutex(){
  int sum=0;
  struct timeval begin,end;
  std::mutex mtx;
  gettimeofday(&begin,NULL);
#  pragma omp parallel for
  for(int i=0;i<limit;++i){
    std::lock_guard<std::mutex> lk(mtx);
    ++sum;
  }
  gettimeofday(&end,NULL);
  return (pair){duration_time(&begin,&end), __func__};
}

pair red(){
  int sum=0;
  struct timeval begin,end;
  gettimeofday(&begin,NULL);
# pragma omp parallel for reduction(+: sum)
  for(int i=0;i<limit;++i)
    ++sum;
  gettimeofday(&end,NULL);
  return (pair){duration_time(&begin,&end), __func__};
}

pair atomic(){
  std::atomic<int> sum=0;
  struct timeval begin,end;
  gettimeofday(&begin,NULL);
#  pragma omp parallel for
  for(int i=0;i<limit;++i)
    sum.fetch_add(1);
  gettimeofday(&end,NULL);
  return (pair){duration_time(&begin,&end), __func__};
}

int main(){
  constexpr int func_num=5;
  int thread_num=get_thread_num();
  pair (*func[func_num])(void) ={seq, cri, cri_mutex, red, atomic};
  for(int i=0;i<func_num;++i){
    pair result=func[i]();
    printf("%d threads %s: %lf sec.\n", thread_num, result.name, result.time);
  }
}
