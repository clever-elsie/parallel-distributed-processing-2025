#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

__global__ void Copy(float*A,float*B){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  B[index]=A[index];
}

__global__ void Scala(float*A,float*B,float scala){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  B[index]=A[index]*scala;
}

__global__ void Add(float*A,float*B,float*C){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  C[index]=A[index]+B[index];
}

__global__ void Triad(float*A,float*B,float*C,float scala){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  C[index]=A[index]*scala+B[index];
}

#define ex(name) #name, name
constexpr dim3 block(1024), thread(1024);
constexpr int n=1024;
constexpr float scala=2;
float *m[3], *dm[3];

template<class F,class... Args>
void check(const char* name, F f, Args... args){
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaMemcpy(dm[0], m[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dm[1], m[1], n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(start, 0);
  f<<<block,thread>>>(args...);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaMemcpy(m[0], dm[0], n*n*sizeof(float), cudaMemcpyDeviceToHost);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cout<<name<<": "<<time<<" ms"<<std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

int main(){
  for(int i=0;i<3;++i){
    m[i]=new float[n*n];
    cudaMalloc(&dm[i], n*n*sizeof(float));
    for(int j=0;j<n*n;++j)
      m[i][j]=rand()%100;
  }
  
  Copy<<<1,1>>>(dm[0], dm[1]); // warmup
  check(ex(Copy), dm[0], dm[1]);
  check(ex(Scala), dm[0], dm[1], scala);
  check(ex(Add), dm[0], dm[1], dm[2]);
  check(ex(Triad), dm[0], dm[1], dm[2], scala);


  for(int i=0;i<3;++i){
    cudaFree(dm[i]);
    delete[] m[i];
  }
  return 0;
}