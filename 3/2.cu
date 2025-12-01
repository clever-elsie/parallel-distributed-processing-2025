#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <bit>
#include <vector>
#include <climits>
#include <cuda.h>

#include "0.cuh"

using namespace std;
constexpr int N=4000;
__global__ void max_val(int*array, int*max){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int s_max[32];
	if(threadIdx.x<32)
		s_max[threadIdx.x]=INT_MIN;
	__syncthreads();
	if(index>=N) return;
	atomicMax(&s_max[threadIdx.x&31], array[index]);
	__syncthreads();
	if(threadIdx.x<32)
		for(int i=1;i<32;i<<=1)
			if((threadIdx.x&(i<<1)-1)==0)
				atomicMax(&s_max[threadIdx.x], s_max[threadIdx.x+i]);
	if(threadIdx.x==0)
		atomicMax(max,s_max[0]);
}
__global__ void max_idx_with_max_val(int*array, int*max, int*idx){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>=N) return;
	if(array[index]==*max)
		atomicMin(idx, index);
}

__global__ void max_idx(int*array, int*max_idx){
  const int index=threadIdx.x;
  const int loop=(N+blockDim.x-1)/blockDim.x;
  __shared__ int s_max[32];
  if(threadIdx.x<32)
    s_max[threadIdx.x]=INT_MIN;
  __syncthreads();
  for(int i=0;i<loop;++i){
    const int idx=index+i*blockDim.x;
    if(idx<N)
      atomicMax(&s_max[threadIdx.x&31], array[idx]);
  }
  __syncthreads();
  if(threadIdx.x<32)
		#if 1
		for(int i=1;i<32;i<<=1)
			if((threadIdx.x&(i<<1)-1)==0)
				atomicMax(&s_max[threadIdx.x], s_max[threadIdx.x+i]);
		#else
			atomicMax(&s_max[0], s_max[threadIdx.x]);
		#endif
  __syncthreads();
  const int max=s_max[0];
	int maxidx=N;
  for(int i=loop-1;i>=0;--i){
    const int idx=index+i*blockDim.x;
    if(idx<N && array[idx]==max)
			maxidx=idx;
  }
	if(maxidx!=N)
		atomicMin(max_idx,maxidx);
}

int main(){
  vector<int> array(N);
  int ans_id_from_gpu=-1;
  array[0]=0;
  for(int i=1;i<N;++i)
    array[i]=array[i-1]+(rand()&3)+1;
  for(int i=N-1;i>0;--i)
    swap(array[rand()%i], array[i]); 
  cuda_memory<int> g_array(N);
  cuda_memory<int> g_max_idx(1);
  g_array.push(array);
  g_max_idx.push(N);

	#if 1
  elapsed_time et;
  constexpr dim3 block(1), thread(1024);
  cout<<"Time: "<<et({block, thread}, max_idx, g_array.ptr(), g_max_idx.ptr())<<" ms"<<endl;
	#else
	elapsed_time et[2];
	constexpr dim3 block((N+1023/1024)), thread(1024);
	cuda_memory<int> g_max(1);
	g_max.set(INT_MIN);
	auto tm=et[0].get_future({block,thread}, max_val, g_array.ptr(), g_max.ptr());
	auto ti=et[1].get_future({block,thread}, max_idx_with_max_val, g_array, g_max.ptr(), g_max_idx.ptr());
	cout<<"Time: "<<tm.get()+ti.get()<<" ms"<<endl;
	#endif

  g_max_idx.pop(ans_id_from_gpu);

  ptrdiff_t ans_id=max_element(array.begin(), array.end())-array.begin();
  cout<<(ans_id==ans_id_from_gpu?"Valid":"Invalid")<<endl;

  return 0;
}