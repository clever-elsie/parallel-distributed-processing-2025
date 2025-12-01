#include <stdio.h>
#include <omp.h>
#include <condition_variable>
#include <mutex>

int main(){
  int cur=0;
  std::mutex mtx;
  std::condition_variable cv;
# pragma omp parallel
  {
    const int id=omp_get_thread_num();
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]{ return cur==id; });
    printf("My Thread ID is %d.\n",id);
    ++cur;
    cv.notify_all();
  }
}