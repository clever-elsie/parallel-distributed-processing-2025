#include <stdio.h>
#include <omp.h>

int main(int argc, char**argv){
  int cur=0;
# pragma omp parallel
  {
    int id=omp_get_thread_num();
    while(cur!=id);
    printf("My Thread ID is %d.\n",omp_get_thread_num());
    cur+=1;
  }
  return 0;
}