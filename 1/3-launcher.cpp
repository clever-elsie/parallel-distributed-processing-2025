#include <stdio.h>
#include <cstdlib>
#include <string>

using namespace std;

constexpr const char* execmode[]={
  "reduction",
  "manual_reduction_with_atomic",
  "atomic_vector",
  "std::mutex",
  "pthread_mutex"
};

int main(){
  for(int i=0;i<sizeof(execmode)/sizeof(execmode[0]);++i)
    for(int j=1;j<=4;j<<=1){
      int ret=system(
        ("OMP_NUM_THREADS=" + to_string(j)
         + " ./3.out " + string(execmode[i])).c_str());
      if(ret!=0){
        printf("Execution Failed for %s with %d threads\n", execmode[i], j);
        return 1;
      }
    }
  return 0;
}