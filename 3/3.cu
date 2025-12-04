#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <queue>
#include <functional>
#include <cuda.h>
#include "0.cuh"
using namespace std;

constexpr int ARRAYS=64;

__global__ void L2Norm(int Elems, float*inputs, float*out){
  const int index=threadIdx.x;
  __shared__ float s_sum[32];
  if(threadIdx.x<32) s_sum[threadIdx.x]=0.0f;
  __syncthreads();
  const int loop=(Elems+blockDim.x-1)/blockDim.x;
  for(int i=0;i<loop;++i){
    const int idx=blockDim.x*i + index;
    if(idx<Elems)
      atomicAdd(&s_sum[threadIdx.x&31], inputs[idx]*inputs[idx]);
  }
  __syncthreads();
  if(threadIdx.x<32)
    for(int i=1;i<32;i<<=1)
      if((threadIdx.x&(i<<1)-1)==0)
        atomicAdd(&s_sum[threadIdx.x], s_sum[threadIdx.x+i]);
  __syncthreads();
  if(threadIdx.x==0)
    *out=sqrt(s_sum[0]);
}

void L2Norm_cpu(int Elems, float*inputs, float*out){
  priority_queue<float,vector<float>,greater<float>> q;
  for(int i=0;i<Elems;++i)
    q.push(inputs[i]*inputs[i]);
  while(q.size()>1){
    float a=q.top();
    q.pop();
    float b=q.top();
    q.pop();
    q.push(a+b);
  }
  *out=sqrt(q.top());
}

int main(){
  int num_array=ARRAYS;
  vector<int> Elems(num_array);
  int total_elems=0;
  for(auto&e:Elems)
    total_elems+=(e=rand()&2048-1);
  vector<float> inputs(total_elems);
  for(auto&i:inputs)
    i=(float)(rand()&255);
  vector<float> out(num_array);
  cuda_memory<int> g_Elems(num_array);
  cuda_memory<float> g_inputs(total_elems), g_out(num_array);
  g_inputs.push(inputs);
  g_Elems.push(Elems);
  g_out.set(0.0f);
  
  vector<elapsed_time> et(num_array);
  vector<std::future<float>> futures(num_array);
  vector<cudaStream_t> streams(num_array);
  int offset=0;
  for(int i=0;i<num_array;++i){
    cudaStreamCreate(&streams[i]);
    int elems=Elems[i];
    float* inputs=g_inputs.ptr()+offset;
    elapsed_time::cuda_call_args cargs(1, 1024, 0, streams[i]);
    futures[i]=et[i].get_future(cargs, L2Norm, elems, inputs, g_out.ptr()+i);
    offset+=elems;
  }

  vector<float> out_cpu(num_array);
  offset=0;
  for(int i=0;i<num_array;offset+=Elems[i++])
    L2Norm_cpu(Elems[i], inputs.data()+offset, out_cpu.data()+i);

  float time=0;
  for(int i=0;i<num_array;++i)
    time+=futures[i].get();
  cout<<"Time: "<<time<<" ms"<<endl;
  g_out.pop(out);

  bool is_ok=true;
  for(int i=0;i<num_array;++i)
    if(
      fabs(out[i]-out_cpu[i])/max(out[i],out_cpu[i])>0.005f
      && fabs(out[i]-out_cpu[i])>0.1){
      is_ok=false;
      cout<<"out["<<i<<"]="<<out[i]<<" != out_cpu["<<i<<"]="<<out_cpu[i]<<endl;
      break;
    }
  cout<<(is_ok?"OK":"NG")<<endl;
  return 0;
}