#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <string_view>
#include <mutex>
#include <pthread.h>
#include <atomic>
#include <vector>
#include <numeric>
#include "0.h"

constexpr int million=1000*1000;

enum class execmode_t{
  reduction,
  manual_reduction_with_atomic,
  atomic_vector,
  std_mutex,
  pthread_mutex
};

execmode_t which(int argc, char**argv){
  constexpr std::string_view s[]={
    "reduction",
    "manual_reduction_with_atomic",
    "atomic_vector",
    "std::mutex",
    "pthread_mutex"
  };
  if(argc>1)
    for(int i=0;i<sizeof(s)/sizeof(s[0]);++i)
      if(s[i]==argv[1]){
        printf("execmode: %s\n", s[i].data());
        return static_cast<execmode_t>(i);
      }
  printf("execmode: reduction\n");
  return execmode_t::reduction;
}

int main(int argc, char**argv){
  const execmode_t execmode=which(argc, argv);

  int* input=(int*)malloc(million*sizeof(int));
  for(int i=0;i<million;++i) input[i]=rand()&(4096-1);
  int* histogram=(int*)calloc(1024,sizeof(int));

  const int thread_num=get_thread_num(); // manual_reduction_with_atomic
  const int unit=(million+thread_num-1)/thread_num; // ceil // manual_reduction_with_atomic

  auto atomic_histogram=new std::atomic<int>[1024]; // atomic_vector
  // C++20ではデフォルトコンストラクタが0で初期化するのでnewすればよい

  std::mutex mutex; // std_mutex

  pthread_mutex_t pmutex; // pthread_mutex
  pthread_mutex_init(&pmutex, NULL);

  struct timeval begin,end;
  gettimeofday(&begin,NULL);

  switch(execmode){
    case execmode_t::reduction:
#     pragma omp parallel for reduction(+:histogram[:1024])
      for(int i=0;i<million;++i)
        ++histogram[input[i]>>2];
    break;

    case execmode_t::manual_reduction_with_atomic:
#     pragma omp parallel
      {
        int id=omp_get_thread_num();
        int itr=unit*id;
        int end=itr+unit;
        if(end>million) end=million;
        int *bucket=(int*)calloc(1024,sizeof(int));
        for(;itr<end;++itr)
          ++bucket[input[itr]>>2];
        for(int i=0;i<1024;++i)
#         pragma omp atomic
          histogram[i]+=bucket[i];
      }
    break;
    case execmode_t::atomic_vector:
#     pragma omp parallel for
      for(int i=0;i<million;++i)
        atomic_histogram[input[i]>>2].fetch_add(1);
    break;
    case execmode_t::std_mutex:
#     pragma omp parallel for
      for(int i=0;i<million;++i){
        std::lock_guard lk(mutex);
        ++histogram[input[i]>>2];
      }
    break;
    case execmode_t::pthread_mutex:
#     pragma omp parallel for
      for(int i=0;i<million;++i){
        pthread_mutex_lock(&pmutex);
        ++histogram[input[i]>>2];
        pthread_mutex_unlock(&pmutex);
      }
    break;
  }
  gettimeofday(&end,NULL);
  printf("duration time %lf sec\n",duration_time(&begin,&end));
  
  const int sum=
    execmode==execmode_t::atomic_vector?
      std::accumulate(atomic_histogram,atomic_histogram+1024,0)
      :std::accumulate(histogram,histogram+1024,0);
  printf("sum is %d\n",sum);
  { //destructor
    free(input);
    free(histogram);
    delete[] atomic_histogram;
    pthread_mutex_destroy(&pmutex);
  }
}